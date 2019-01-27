The train.py and eval_ood.py files handle training and testing the OE and baseline models. The data.py file specifies preprocessing. Pre-trained models used in the paper are in the snapshots folder.

**Commands for training/testing models**

PTBC train

python -u train.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 50 --save PTBC.pt --use_OE no --wikitext_char --data data/pennchar  --bptt 150


PTBC test

python eval_ood.py --data data/pennchar --resume PTBC.pt --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --bptt 150 --model QRNN --character_level


PTBC_OE train

python -u train.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs 5 --save PTBC_OE_finetune.pt --use_OE yes --wikitext_char --data data/pennchar --bptt 150 --resume PTBC.pt


PTB models are trained/tested analogously, but without --character_level when running eval_ood.py.


**Models in snapshots folder (not on github due to space limits)**

PTBC.pt : baseline character level model

PTBC_OE_2.pt : WikiText2 OE (fine-tuned)

PTBC_OE_103.pt : WikiText103 OE (fine-tuned)

PTB : baseline word level model

PTB_OE_2 : WikiText2 OE (fine-tuned)

PTB_OE_103 : WikiText103 OE (fine-tuned)
