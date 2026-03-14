import os
import urllib.request

FILES = {
        # YOLOv3 tiny
        "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",

        # YOLOv3 full
        "yolov3.weights": "https://pjreddie.com/media/files/yolov3.weights",
        "yolov3.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",

        # class labels
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
}

MODEL_DIR = "models"


def download_file(filename, url):
        filepath = os.path.join(MODEL_DIR, filename)

        if os.path.exists(filepath):
                print(f"✓ {filename} already exists, skipping.")
                return

        print(f"Downloading {filename}...")

        # put in an agent request or we will hit 403 forbidden errors
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

        with urllib.request.urlopen(request) as response, open(filepath, "wb") as f:
                f.write(response.read())

        print(f"✓ Downloaded {filename}")


def main():
        os.makedirs(MODEL_DIR, exist_ok=True)

        for filename, url in FILES.items():
                download_file(filename, url)

        print("\nAll model files downloaded successfully.")


if __name__ == "__main__":
        main()