#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# This script based on alignment_by_row_channels.py by Allison Deal, see
# https://github.com/allisonnicoledeal/VideoSync/blob/master/alignment_by_row_channels.py
"""
This module contains the detector class for knowing the offset
difference for audio and video files, containing audio recordings
from the same event. It relies on ffmpeg being installed and
the python libraries scipy and numpy.
"""
from __future__ import unicode_literals
from __future__ import absolute_import

_doc_template = '''
    %(prog)s <file1> <file2>

This program reports the offset difference for audio and video files,
containing audio recordings from the same event. It relies on ffmpeg being
installed and the python libraries scipy and numpy.

It reports back the offset. Example:

    %(prog)s good_video_shitty_audio.mp4 good_audio_shitty_video.mp4

    Result: The beginning of good_video_shitty_audio.mp4 needs to be trimmed off 11.348 seconds for all files to be in sync

'''
import os
import sys
from collections import defaultdict
import tempfile
import shutil
import logging

import numpy as np

from . import communicate
from .utils import check_and_decode_filenames
from . import _cache


__all__ = [
    'SyncDetector',
    'main',
    ]

_logger = logging.getLogger(__name__)


def _mk_freq_trans_summary(data, fft_bin_size, overlap, box_height, box_width, maxes_per_box):
    """
    Return characteristic frequency transition's summary.

    The dictionaries to be returned are as follows:
    * key: The frequency appearing as a peak in any time zone.
    * value: A list of the times at which specific frequencies occurred.
    """
    freqs_dict = defaultdict(list)

    boxes = defaultdict(list)
    for x, j in enumerate(range(int(-overlap), len(data), int(fft_bin_size - overlap))):
        sample_data = data[max(0, j):max(0, j) + fft_bin_size]
        if (len(sample_data) == fft_bin_size):  # if there are enough audio points left to create a full fft bin
            intensities = np.abs(np.fft.fft(sample_data))  # intensities is list of fft results
            box_x = x // box_width
            for y in range(len(intensities) // 2):
                box_y = y // box_height
                # x: corresponding to time
                # y: corresponding to freq
                boxes[(box_x, box_y)].append((intensities[y], x, y))
                if len(boxes[(box_x, box_y)]) > maxes_per_box:
                    boxes[(box_x, box_y)].remove(min(boxes[(box_x, box_y)]))
    #
    for box_x, box_y in list(boxes.keys()):
        for intensity, x, y in boxes[(box_x, box_y)]:
            freqs_dict[y].append(x)

    del boxes
    return freqs_dict


def _find_delay(freqs_dict_orig, freqs_dict_sample):
    keys = set(freqs_dict_sample.keys()) & set(freqs_dict_orig.keys())
    #
    if not keys:
        raise Exception(
            """I could not find a match. Consider giving a large value to \
"max_misalignment" if the target medias are sure to shoot the same event.""")
    #
    t_diffs = defaultdict(int)
    for key in keys:
        for x_i in freqs_dict_sample[key]:  # determine time offset
            for x_j in freqs_dict_orig[key]:
                delta_t = x_i - x_j
                t_diffs[delta_t] += 1

    t_diffs_sorted = sorted(list(t_diffs.items()), key=lambda x: x[1])
    # _logger.debug(t_diffs_sorted)
    time_delay = t_diffs_sorted[-1][0]

    return time_delay


class SyncDetector(object):
    def __init__(self, sample_rate=48000, dont_cache=False):
        self._working_dir = tempfile.mkdtemp()
        self._sample_rate = sample_rate
        self._dont_cache = dont_cache
        self._orig_infos = {}  # per filename

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        retry = 3
        while retry > 0:
            try:
                shutil.rmtree(self._working_dir)
                break
            except:
                import time
                retry -= 1
                time.sleep(1)

    def _extract_audio(self, sample_rate, video_file, starttime_offset, duration):
        """
        Extract audio from video file, save as wav auido file

        INPUT: Video file, and its index of input file list
        OUTPUT: Does not return any values, but saves audio as wav file
        """
        return communicate.media_to_mono_wave(
            video_file, self._working_dir,
            starttime_offset=starttime_offset,
            duration=duration,
            sample_rate=sample_rate)

    def _get_media_info(self, fn):
        if fn not in self._orig_infos:
            self._orig_infos[fn] = communicate.get_media_info(fn)
        return self._orig_infos[fn]

    def _align(self, sample_rate, files, fft_bin_size, overlap, box_height, box_width, samples_per_box,
               max_misalignment, known_delay_ge_map):
        """
        Find time delays between video files
        """
        def _each(idx):
            exaud_args = dict(
                sample_rate=sample_rate, video_file=files[idx],
                starttime_offset=known_delay_ge_map.get(idx, 0),
                duration=max_misalignment * 2)
            # First, try getting from cache.
            ck = None
            if not self._dont_cache:
                for_cache = dict(exaud_args)
                for_cache.update(dict(
                        fft_bin_size=fft_bin_size,
                        overlap=overlap,
                        box_height=box_height,
                        box_width=box_width,
                        samples_per_box=samples_per_box,
                        atime=os.path.getatime(files[idx])
                        ))
                ck = _cache.make_cache_key(**for_cache)
                cv = _cache.get("_align", ck)
                if cv:
                    return cv
            else:
                _cache.clean("_align")

            # Not found in cache.
            wavfile = self._extract_audio(**exaud_args)
            raw_audio, rate = communicate.read_audio(wavfile)
            ft_dict = _mk_freq_trans_summary(
                raw_audio,
                fft_bin_size, overlap,
                box_height, box_width, samples_per_box)  # bins, overlap, box height, box width
            del raw_audio
            if not self._dont_cache:
                _cache.set("_align", ck, (rate, ft_dict))
            return rate, ft_dict
        #
        tmp_result = [0.0]

        # Process first file
        rate, ft_dict1 = _each(0)
        for i in range(len(files) - 1):
            # Process second file
            rate, ft_dict2 = _each(i + 1)

            # Determie time delay
            delay = _find_delay(ft_dict1, ft_dict2)
            samples_per_sec = float(rate) / float(fft_bin_size)
            seconds = float(delay) / float(samples_per_sec)

            #
            tmp_result.append(-seconds)

        result = np.array(tmp_result)
        if known_delay_ge_map:
            for i in range(len(result)):
                if i in known_delay_ge_map:
                    result += known_delay_ge_map[i]
                    result[i] -= known_delay_ge_map[i]

        # build result
        pad_pre = result - result.min()
        trim_pre = -(pad_pre - pad_pre.max())
        #
        return pad_pre, trim_pre

    def align(self, files, fft_bin_size=1024, overlap=0, box_height=512, box_width=43, samples_per_box=7,
              max_misalignment=0, known_delay_ge_map={}):
        """
        Find time delays between video files
        """
        # First, try finding delays roughly by passing low sample rate.
        pad_pre, trim_pre = self._align(
            44100 // 4, files, fft_bin_size, overlap, box_height, box_width, samples_per_box,
            max_misalignment, known_delay_ge_map)

        # update knwown map, and max_misalignment
        known_delay_ge_map = {
            i: max(0.0, int(trim_pre[i] - 5.0))
            for i in range(len(trim_pre))
            }
        max_misalignment = 30

        # Finally, try finding delays precicely
        pad_pre, trim_pre = self._align(
            self._sample_rate, files, fft_bin_size, overlap, box_height, box_width, samples_per_box,
            max_misalignment, known_delay_ge_map)
        #
        infos = [self._get_media_info(fn) for fn in files]
        orig_dur = np.array([inf["duration"] for inf in infos])
        strms_info = [
            (inf["streams"], inf["streams_summary"]) for inf in infos]
        pad_post = list(
            (pad_pre + orig_dur).max() - (pad_pre + orig_dur))
        trim_post = list(
            (orig_dur - trim_pre) - (orig_dur - trim_pre).min())
        #
        return [{
                "trim": trim_pre[i],
                "pad": pad_pre[i],
                "orig_duration": orig_dur[i],
                "trim_post": trim_post[i],
                "pad_post": pad_post[i],
                "orig_streams": strms_info[i][0],
                "orig_streams_summary": strms_info[i][1],
                }
                for i in range(len(files))]

    @staticmethod
    def summarize_stream_infos(result_from_align):
        """
        This is a service function that calculates several summaries on
        information about streams of all medias returned by
        SyncDetector#align.

        Even if "align" has only detectable delay information, you are
        often in trouble. This is because editing for lineup of targeted
        plural media involves unification of sampling rates (etc) in many
        cases.

        Therefore, this function calculates the maximum sampling rate etc.
        through all files, and returns it in a dictionary format.
        """
        result = dict(
            max_width=0,
            max_height=0,
            max_sample_rate=0,
            max_fps=0.0,
            has_video = [],
            has_audio = [])
        for ares in result_from_align:
            summary = ares["orig_streams_summary"]  # per single media

            result["max_width"] = max(
                result["max_width"], summary["max_resol_width"])
            result["max_height"] = max(
                result["max_height"], summary["max_resol_height"])
            result["max_sample_rate"] = max(
                result["max_sample_rate"], summary["max_sample_rate"])
            result["max_fps"] = max(
                result["max_fps"], summary["max_fps"])

            result["has_video"].append(
                summary["num_video_streams"] > 0)
            result["has_audio"].append(
                summary["num_audio_streams"] > 0)
        return result


def _bailout(parser):
    parser.print_usage()
    sys.exit(1)


def main(args=sys.argv):
    import argparse
    import json

    parser = argparse.ArgumentParser(prog=args[0], usage=_doc_template)
    parser.add_argument(
        '--max_misalignment',
        type=str, default="600",
        help='When handling media files with long playback time, \
it may take a huge amount of time and huge memory. \
In such a case, by changing this value to a small value, \
it is possible to indicate the scanning range of the media file to the program. \
(default: %(default)s)')
    parser.add_argument(
        '--known_delay_ge_map',
        type=str,
        help='''When handling media files with long playback time, \
furthermore, when the delay time of a certain movie is large,
it may take a huge amount of time and huge memory. \
In such a case, you can give a mapping of the delay times that are roughly known. \
Please pass it in JSON format, like '{"1": 120}'. The key is an index corresponding \
to the file passed as "file_names". The value is the number of seconds, meaning \
"at least larger than this".''')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=48000,
        help='''In this program, delay is examined by unifying all the sample rates \
of media files into the same one. If this value is the value itself of the media file \
itself, the result will be more precise. However, this wastes a lot of memory, so you \
can reduce memory consumption by downsampling (instead losing accuracy a bit). \
The default value uses quite a lot of memory, but if it changes to a value of, for example, \
44100, 22050, etc., although a large error of about several tens of milliseconds \
increases, the processing time is greatly shortened. (default: %(default)d)''')
    parser.add_argument(
        '--dont_cache',
        action="store_true",
        help='''Normally, this script stores the result in cache ("%s"). \
If you hate this behaviour, specify this option.''' % (
            _cache.cache_root_dir))
    parser.add_argument(
        '--json',
        action="store_true",
        help='To report in json format.',)
    parser.add_argument(
        'file_names',
        nargs="+",
        help='Media files including audio streams. \
It is possible to pass any media that ffmpeg can handle.',)
    args = parser.parse_args(args[1:])
    known_delay_ge_map = {}
    if args.known_delay_ge_map:
        known_delay_ge_map = json.loads(args.known_delay_ge_map)
        known_delay_ge_map = {
            int(k): known_delay_ge_map[k]
            for k in known_delay_ge_map.keys()
            }

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stderr,
        format="%(created)f|%(levelname)5s:%(module)s#%(funcName)s:%(message)s")

    file_specs = check_and_decode_filenames(
        args.file_names, min_num_files=2)
    if not file_specs:
        _bailout(parser)

    with SyncDetector(
        sample_rate=args.sample_rate,
        dont_cache=args.dont_cache) as det:
        result = det.align(
            file_specs,
            max_misalignment=communicate.parse_time(args.max_misalignment),
            known_delay_ge_map=known_delay_ge_map)
    if args.json:
        print(json.dumps(
                {'edit_list': list(zip(file_specs, result))}, indent=4, sort_keys=True))
    else:
        report = []
        for i, path in enumerate(file_specs):
            if not (result[i]["trim"] > 0):
                continue
            report.append(
                """Result: The beginning of '%s' needs to be trimmed off %.4f seconds \
(or to be added %.4f seconds padding) for all files to be in sync""" % (
                    path, result[i]["trim"], result[i]["pad"]))
        if report:
            print("\n".join(report))
        else:
            print("files are in sync already")


if __name__ == "__main__":
    main()
