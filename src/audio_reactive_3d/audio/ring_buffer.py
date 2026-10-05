"""Lock-free single-producer/single-consumer ring buffer for audio frames.

Designed for exactly one producer thread (audio capture callback) and one
consumer thread (DSP worker). The producer only ever writes and advances
``_write_index``; the consumer only ever reads and advances ``_read_index``.
Both indices are plain Python ints protected by the GIL for their individual
read/modify/write -- no locks, no allocations, on the audio callback path.

The buffer is a fixed-size preallocated numpy array (``capacity`` frames x
``channels``). ``write()`` never blocks and never raises on overflow; instead
it drops the oldest unread samples (overwrites them), which is the right
tradeoff for real-time audio: we must never stall the capture callback.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    """Fixed-capacity, allocation-free ring buffer of float32 audio frames.

    Parameters
    ----------
    capacity:
        Number of frames (samples per channel) the buffer can hold.
    channels:
        Number of interleaved channels per frame.
    """

    def __init__(self, capacity: int, channels: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        self._capacity = capacity
        self._channels = channels
        self._data = np.zeros((capacity, channels), dtype=np.float32)
        self._write_index = 0
        self._read_index = 0
        # Number of frames currently available to read. Tracked separately
        # from the indices so we can distinguish "empty" from "full" (both
        # of which would otherwise have write_index == read_index).
        self._available = 0

    @property
    def capacity(self) -> int:
        """Total number of frames the buffer can hold."""
        return self._capacity

    @property
    def available(self) -> int:
        """Number of frames currently available to read."""
        return self._available

    def write(self, frames: np.ndarray) -> None:
        """Push ``frames`` (shape ``(n, channels)``) into the buffer.

        Allocation-free: all work is done via preallocated-array slicing.
        If ``frames`` is larger than the remaining free space, the oldest
        unread samples are overwritten (never blocks, never raises).
        """
        n = frames.shape[0]
        if n == 0:
            return
        if n >= self._capacity:
            # Only the most recent `capacity` frames can possibly fit.
            frames = frames[-self._capacity :]
            n = self._capacity

        end = self._write_index + n
        if end <= self._capacity:
            self._data[self._write_index : end] = frames
        else:
            first_part = self._capacity - self._write_index
            self._data[self._write_index :] = frames[:first_part]
            self._data[: end - self._capacity] = frames[first_part:]

        self._write_index = end % self._capacity

        # Advance read_index too if we've overwritten unread data (overflow).
        overflow = n - (self._capacity - self._available)
        if overflow > 0:
            self._read_index = (self._read_index + overflow) % self._capacity
            self._available = self._capacity
        else:
            self._available += n

    def read(self, n: int) -> np.ndarray | None:
        """Pop up to ``n`` frames from the buffer.

        Returns a new ``(m, channels)`` array with ``m <= n`` (``m`` may be
        less than ``n`` if fewer frames are available), or ``None`` if the
        buffer is empty.
        """
        if self._available == 0:
            return None

        n = min(n, self._available)
        start = self._read_index
        end = start + n
        if end <= self._capacity:
            out = self._data[start:end].copy()
        else:
            first_part = self._capacity - start
            out = np.empty((n, self._channels), dtype=np.float32)
            out[:first_part] = self._data[start:]
            out[first_part:] = self._data[: end - self._capacity]

        self._read_index = end % self._capacity
        self._available -= n
        return out

    def peek_latest(self, n: int) -> np.ndarray | None:
        """Return (without consuming) the most recent ``n`` written frames.

        Useful for a DSP worker that wants the latest window for an FFT
        without needing to track a separate read cursor. Returns ``None``
        if fewer than ``n`` frames have ever been written.
        """
        if self._available < n and self._write_index < n and self._available < self._capacity:
            # Not enough history yet.
            if self._available < n:
                return None

        n = min(n, self._capacity)
        end = self._write_index
        start = end - n
        if start >= 0:
            return self._data[start:end].copy()

        out = np.empty((n, self._channels), dtype=np.float32)
        first_part = -start
        out[:first_part] = self._data[self._capacity - first_part :]
        out[first_part:] = self._data[:end]
        return out
