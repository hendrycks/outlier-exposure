This code is used to create the complementary 64x64 ImageNet images from the 800 classes that complement Tiny ImageNet. We use this code rather than
rescaling ImageNet images because Tiny ImageNet images are cropped using bounding box information when available. Future experiments could simply involve
rescaling and center-croping ImageNet images, but that would take them slightly farther from the Tiny ImageNet distribution.

