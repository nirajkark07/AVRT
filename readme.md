
# Live 3D Reconstruction Demos

## Table of Contents

- [Description](#Description)
- [Installation](#Installation)
- [Running the Project](#running-the-project)
- [Demo](#Demo)

## Description
This project showcases my work with the AVRT project. The goal of this project is to develop an AR assisted maintenance tool on the Hololens2 using pose estimation (FoundationPose) and object detection (YoloV10) algorithms.

## Installation
1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/yourproject.git
```
   
2. **Pull Docker File (FoundationPose Instructions)**:

If first time running project, build all extensions.

```bash
bash build_all.sh
```

Pull and run docker container.

```bash
cd docker/
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose  # Or to build from scratch: docker build --network host -t foundationpose .
bash docker/run_container_modified.sh
```

3. **Open Unreal Engine Project File**:

In Unreal Projects folder, open AVRT.uproject.

Finish this up later.


## Demo
Below is a current demo of the project.

https://github.com/user-attachments/assets/8fa6a210-7391-4c8d-839d-6a41c9e86395


---

## Contact
This project showcases my ongoing work with VeyondMetaverse. For further details or inquiries, feel free to reach out to me at nkarki@torontomu.ca.


