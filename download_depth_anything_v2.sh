wget -P ./pretrained-models/depth-anything-v2 https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth

git clone https://github.com/DepthAnything/Depth-Anything-V2
mv Depth-Anything-V2/depth_anything_v2 ./
rm -rf Depth-Anything-V2