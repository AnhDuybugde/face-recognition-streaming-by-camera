"""Display frames from the default local webcam."""

import cv2

from src.capture.video_source import VideoSource


def main() -> None:
    """Run the webcam display loop until the user presses ``q``."""
    source = None
    try:
        source = VideoSource(0)
        print(f"Webcam width: {source.width:.0f}")
        print(f"Webcam height: {source.height:.0f}")
        print(f"Reported webcam FPS: {source.fps:.2f}")

        while True:
            frame = source.read()
            if frame is None:
                continue

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if source is not None:
            source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
