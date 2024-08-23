# Aman Chadha / aman@amanchadha.com
# www.amanchadha.com / www.aman.ai

# extract all the zip'd pre-trained models and datasets
find . -name "*.zip" | while read filename; do unzip -o -d "`dirname "$filename"`" "$filename"; done;

# get imagenet-vgg-verydeep-19.mat for Course 4 "Convolutional Neural Networks" Week 4 assignment: "Neural Style Transfer"
curl https://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat > "./C4 - Convolutional Neural Networks/Week 4/Neural Style Transfer/pretrained-model/imagenet-vgg-verydeep-19.mat"
