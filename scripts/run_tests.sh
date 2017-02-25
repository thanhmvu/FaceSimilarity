#!/bin/sh

python test_vggface_firstexp.py 'feret' 'initial.ckpt'
python test_vggface_firstexp.py 'feret' '2nd_fine_tune_norm_2fc.ckpt'
python test_vggface_firstexp.py 'feret' '2nd_continue_norm_all.ckpt'
python test_vggface_firstexp.py 'feret' '2nd_reinitialize_norm_2fc.ckpt'
python test_vggface_firstexp.py '10k' 'initial.ckpt'
python test_vggface_firstexp.py '10k' '2nd_fine_tune_norm_2fc.ckpt'
python test_vggface_firstexp.py '10k' '2nd_continue_norm_all.ckpt'
python test_vggface_firstexp.py '10k' '2nd_reinitialize_norm_2fc.ckpt'