# YOLO car detector: AutoHeal demo target

This small target project detects the COCO `car` class (class ID 2) in one image. Its baseline uses the older Ultralytics `8.0.196` package and `yolov8n.pt` weights. The tests are deliberately lightweight: they check the model selection and input validation without downloading PyTorch or model weights.

## Run the detector

Use Python 3.10 or 3.11 for this older baseline, then install the pinned dependency. PyTorch and the model weights can take time and disk space to download.

```powershell
cd examples/yolo-car-detector
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py C:\path\to\car-photo.jpg
```

On first inference, Ultralytics downloads `yolov8n.pt`. Annotated images are saved in `runs/detect/cars/`.

Run the lightweight tests from the AutoHeal project environment (which already has pytest installed):

```powershell
cd ..\..
.\.venv\Scripts\python.exe -m pytest examples/yolo-car-detector/tests -q
```

If this AutoHeal checkout uses `.venv\bin` instead of `.venv\Scripts`, use that interpreter path.

## Publish the target repository

AutoHeal requires its target repository to be outside the AutoHeal project directory. Copy this example to your Documents folder, then initialize and push that copy. In GitHub, first create an empty repository named `yolo-car-detector`.

From the AutoHeal project root:

```powershell
Copy-Item -Recurse -Force .\examples\yolo-car-detector "$env:USERPROFILE\Documents\yolo-car-detector"
```

Then, from the copied folder, replace `YOUR-USERNAME` with your GitHub account and run:

```powershell
git init -b main
git add app.py requirements.txt tests .gitignore README.md
git commit -m "Add YOLOv8 car detector baseline"
git remote add origin https://github.com/YOUR-USERNAME/yolo-car-detector.git
git push -u origin main
```

Create a baseline GitHub release with tag `8.0.196` if you want the release history to show the starting version. AutoHeal does not need a baseline release; its `current_version` config value supplies the starting version.

## Prepare a YOLO11 upgrade release

This demo upgrades YOLOv8 to YOLO11 using Ultralytics package version `8.3.0`. The official Ultralytics assets release `v8.3.0` introduced the YOLO11 models, and the documented nano detection weight filename is `yolo11n.pt`.

Download the official weight file from the [Ultralytics assets v8.3.0 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). In your **yolo-car-detector** repository on GitHub, choose **Releases → Draft a new release**, create tag `8.3.0` from `main`, set a title such as `Upgrade detector to YOLO11`, attach the downloaded file named exactly `yolo11n.pt`, and publish the release. GitHub releases support attaching binary assets to a tagged release.

The release tag is intentionally `8.3.0` with no leading `v`: AutoHeal uses it to replace `ultralytics==8.0.196` with `ultralytics==8.3.0`. The asset name updates the model filename in `app.py`, and AutoHeal saves the downloaded weights to `models/yolo11n.pt` in its upgrade branch.

For a later upgrade, publish a greater numeric tag matching the Ultralytics package version and attach one nano detection weight with a name such as `yolo26n.pt`. The included regexes match the current nano model name and semantic package version, and AutoHeal remembers the last successful tag in its state file. The GitHub release API supplies the latest release; AutoHeal does not inspect PyPI or choose a package version independently.

## Connect the demo to AutoHeal

From the AutoHeal CI project root, copy `autoheal.yolo.example.json` to `autoheal.yolo.json`, then replace `YOUR-USERNAME` with your GitHub account. Its target path uses `$USERPROFILE/Documents/yolo-car-detector`; change it if you copied the target elsewhere. The target must be its own clean Git repository with the initial commit and `origin` remote set.

```powershell
Copy-Item autoheal.yolo.example.json autoheal.yolo.json
notepad autoheal.yolo.json
if (Test-Path .\.venv\Scripts\Activate.ps1) { .\.venv\Scripts\Activate.ps1 } else { .\.venv\bin\Activate.ps1 }
python -m autoheal --config autoheal.yolo.json validate
python -m autoheal --config autoheal.yolo.json check
python -m autoheal --config autoheal.yolo.json run --allow-local-tests
```

Set `pull_request.enabled` to `true` in the demo config only after authenticating GitHub CLI with `gh auth login` and setting the correct `pull_request.base_branch`. The default is false; without it, AutoHeal makes a local branch and does not push it.

After a successful run, review the generated branch and its report. Install the updated dependency in the target project's own environment before running the detector with the newly downloaded local weights.

## What AutoHeal changes

- `app.py`: `MODEL_NAME` changes from `yolov8n.pt` to the release asset `yolo11n.pt`.
- `requirements.txt`: the pin changes from `ultralytics==8.0.196` to `ultralytics==8.3.0` based on the release tag.
- `models/yolo11n.pt`: the release asset is downloaded into the target's `models/` directory.

AutoHeal runs the configured tests after these changes. It does not install the target's Python dependencies or run image inference in its test container; these focused tests avoid needing PyTorch, a GPU, or network access.
