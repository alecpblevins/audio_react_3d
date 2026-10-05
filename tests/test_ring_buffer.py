"""Producer/consumer correctness tests for RingBuffer."""

from __future__ import annotations

import numpy as np

from audio_reactive_3d.audio.ring_buffer import RingBuffer


def test_write_then_read_roundtrip() -> None:
    rb = RingBuffer(capacity=16, channels=2)
    frames = np.arange(8 * 2, dtype=np.float32).reshape(8, 2)

    rb.write(frames)

    assert rb.available == 8
    out = rb.read(8)
    assert out is not None
    np.testing.assert_array_equal(out, frames)
    assert rb.available == 0


def test_read_empty_returns_none() -> None:
    rb = RingBuffer(capacity=4, channels=1)
    assert rb.read(4) is None


def test_wraparound_write_and_read() -> None:
    rb = RingBuffer(capacity=10, channels=1)
    first = np.arange(7, dtype=np.float32).reshape(7, 1)
    rb.write(first)
    assert rb.read(7) is not None

    # write_index is now at 7; writing 6 more frames wraps around the end.
    second = np.arange(100, 106, dtype=np.float32).reshape(6, 1)
    rb.write(second)

    out = rb.read(6)
    assert out is not None
    np.testing.assert_array_equal(out, second)


def test_overflow_drops_oldest_samples() -> None:
    rb = RingBuffer(capacity=4, channels=1)
    rb.write(np.array([[1.0], [2.0], [3.0]], dtype=np.float32))
    # Buffer capacity is 4; writing 3 more overflows by 2 frames.
    rb.write(np.array([[4.0], [5.0], [6.0]], dtype=np.float32))

    assert rb.available == 4
    out = rb.read(4)
    assert out is not None
    np.testing.assert_array_equal(out, np.array([[3.0], [4.0], [5.0], [6.0]], dtype=np.float32))


def test_write_larger_than_capacity_keeps_most_recent() -> None:
    rb = RingBuffer(capacity=4, channels=1)
    big = np.arange(10, dtype=np.float32).reshape(10, 1)
    rb.write(big)

    assert rb.available == 4
    out = rb.read(4)
    assert out is not None
    np.testing.assert_array_equal(out, big[-4:])


def test_peek_latest_without_consuming() -> None:
    rb = RingBuffer(capacity=8, channels=1)
    rb.write(np.arange(8, dtype=np.float32).reshape(8, 1))

    peeked = rb.peek_latest(4)
    assert peeked is not None
    np.testing.assert_array_equal(peeked, np.array([[4.0], [5.0], [6.0], [7.0]], dtype=np.float32))
    # peek must not consume: available should be unchanged.
    assert rb.available == 8


def test_peek_latest_insufficient_history_returns_none() -> None:
    rb = RingBuffer(capacity=8, channels=1)
    rb.write(np.arange(3, dtype=np.float32).reshape(3, 1))
    assert rb.peek_latest(4) is None


def test_partial_read_when_fewer_frames_available() -> None:
    rb = RingBuffer(capacity=16, channels=1)
    rb.write(np.arange(3, dtype=np.float32).reshape(3, 1))
    out = rb.read(10)
    assert out is not None
    assert out.shape[0] == 3
