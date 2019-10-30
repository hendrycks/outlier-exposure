## Setup
Follow the steps in the wikitext_reformatted, wmt16, and multi30k directories to use those datasets. Excepting English Web Treebank and Yelp Reviews, torchtext will download other datasets automatically. English Web Treebank must be accessed through the LDC. The Yelp Review data can be found at  
https://drive.google.com/open?id=1U50Q9jnpIUznSaUHS9B8I3vG8xAxtE7z  
After downloading the Yelp Review data, move it to the .data folder.

## Description of code
For experiments:  
train.py trains a model without Outlier Exposure  
train_OE.py trains a model with Outlier Exposure  
eval_OOD_xyz.py runs the OOD evaluation code for xyz as the in-distribution dataset  
  
Other:  
download_wikitext.py is for setting up the wikitext data  
