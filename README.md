This project trains the VGGFace network for face similarity using triplet loss 

## Directories: 

* **scripts:** contains the python scripts to use
* **input:** contains input files
* **description_vectors:** contains the output description vectors 
* **distnace_mat:** contains the distances for all pairs of vectors 
* **distances:**  contains the distances between a list of pairs if using pairs of faces

## Scripts: 

* **compare_scores.py** (deprecated) 
* **crop_datasets.py** uses face detection to crop the images, takes two arguments :
  > arg 1 : Dataset (takes either "10k" or "feret")
  > arg 2 : Method (takes "OpenCV" or "Dlib")
  reads the images from **../DATASET_NAME/** and outputs the cropped images to **../../DATASET_NAME_cropped/**
* **crop_for_mechanicalturk.py:** (deprecated) 
* **format_mechturk_results.py:** Formats the results from the first Mechanical Turk experiment into a CSV file that
  contains a test (two pairs) per line after computing the majoritarian vote and the agreement percentage.
  This format is used by other scripts to make reading results easy. Results from this script are placed in
  **/results/DATASET_NAME_mechturk_formatted.csv** and each line of the file is in this format :
  > first_pair_name, second_pair_name, first_pair_first_image_name, first_pair_second_image_name, second_pair_first_image_name, second_pair_second_image_name, first_pair_score, second_pair_score, agreement_pecentage, computer_vote, human_vote, ("agree" if computer_vote==human_vote else "disagree")
  where "first_pair_name" and "second_pair_name" are unique identifiers of the pair in the following format "first_pair_first_image_name#first_pair_second_image_name" whichever is smaller.
* **format_second_experiment.py** Formats the results from the second Mechanical Turk experiment into a CSV file where
  each line constitutes a triplet (currently it takes the lowest scoring and the highest scoring images to make a triplet, see comments
  inside file for instructions to change that). Results from this script are placed in **/results/second_experiment_formatted.csv** and each line of the file
  has format :
  > anchor_image_name, lowest_scoring_image_name, highest_scoring_image_name
* **generate_outliers_web_page.py** (deprecated)
* **get_feature_vectors.py** Generates a vector file as well a pairwise distance matrix for a specific dataset/face-detection method and trained VGGFace model
  change constants in the beginning of the script as follows :
  > METHOD : either "OpenCV" or "Dlib" (however currently the script does not re-detect faces rather uses the results from **crop_datasets.py**)
  > DATASET_PATH : either "../../feret/" or "../../10k/"
  > MODEL_NAME : a trained model (see section on **Trained Models**)
  > INPUT_FILE : names of all images in the dataset (ex. "all_unique_feret_names.csv")
  > SAVE_IMAGES (deprecated)
* **get_pair_distances.py** (deprecated)
* **h5_to_pysaver.py** (deprecated) Was used to convert the initial trained VGGFace model from the .h5 format to Tensorflow's built-in Saver (required multiple changes in the VGGFace script)
* **make_all_unique_names.py** Reads the Dataset's directory and lists the names of all images in a csv file to be used by other scripts
* **process_brunelle.py** (deprecated) Was used to visualize the scores on the brunelle dataset with a web page and a histogram of bins
* **process_lookalikesdb.py** (deprecated) Was used to visualize the scores on the LookalikesDB dataset with a web page and a histogram of bins
* **process_mechturk_agreement_colormap.py** (deprecated) Was used to generate a colormap of agreement (see http://cs.lafayette.edu/~gharbiw/face-similarity/mech_turk_page/experiment_result.html)
* **process_mechturk_colormap.py** (deprecated) Was used to generate a colormap of confidence (see http://cs.lafayette.edu/~gharbiw/face-similarity/mech_turk_page/experiment_result.html)
* **test_vggface_firstexp.py** Used to test the trained models, takes as arguments:
  > DATASET : the dataset to test on
  > MODEL_NAME : the trained model to use for testing (see section on **Trained Models**)
* **run_tests.sh** Runs the test_vggface_firstexp.py script on multiple models to obtain results at once
* **train_vggface_firstexp.sh** and **train_vggface_quad.sh** (deprecated) were used to train the model using the results from the first vggface experiment (quadruples)
* **train_vggface_triplet.sh** used to train the model using the results from the second vggface experiment (triplet loss)
* **visualize_improvement.sh** generates a web page (in **/results/second_experiment_results/feret_improvement_80.html**) that visualizes the pairs that the computer now recognizes correctly
  (takes two models as input to compare the results from them)
* **visualize_mechturk_statistics.py** (deprecated) Visualizes statistics (about race, gender, facial hair, etc.) about the quadruples presented to testers in the first mechturk experiment
* **visualize_mechturk_statistics_alg_vs_human.py** (deprecated) Visualizes statistics (about race, gender, facial hair, etc.) about the quadruples where the algorithm initially disagreed with testers in the first mechturk experiment
* **run_visualize_mechturk.sh** (deprecated) Was used to run **visualize_mechturk_statistics_alg_vs_human.py** on different agreement percentages
* **visualize_second_experiment.py** Shows the triplets generated by **format_second_experiment.py** in a web page

## Trained Models

Trained models are stored in **scripts/vggface/trainedmodels/**, their description is as follows :

* **initial.ckpt** and **untrained.ckpt** The initial model converted from h5 (untrained)
* **fine_tune_2fc.ckpt** trained using the results from the first Mechanical Turk experiment (quadruples)
* **2nd_fine_tune_norm_2fc.ckpt** trained using the results from the second Mechanical Turk experiment (triplets) - fine tuning the last two layers
* **2nd_continue_norm_all.ckpt** trained using the results from the second Mechanical Turk experiment (triplets) - fine tuning all the layers
* **2nd_reinitialize_norm_2fc.ckpt** trained using the results from the second Mechanical Turk experiment (triplets) - reinitializing the variables on the last two layers then training them

## How to run

To create a webpage of 5000 faces and retrieve their six most similar faces:

1. Move to scripts folder: **cd <proj-root>/scripts/**
2. Filter unreadable images using face detection:   
  a. Provide input parameteres in thanh_init_dataset.py   
  b. Run: **python thanh_init_dataset.py**   
  c. Manually go through the output images and filter out bad faces (poor quality, weird pose, duplicate faces, celebrity faces, ...)  
  d. Redo a, b, c to filter new faces if needed.  
3. The final output should be 100 names &ast; 50 faces/ name = 5000 faces
4. Crop 120x150 images to make spare images: **mogrify -crop 120x120+0+30 &ast;.png**
5. Resize image to 224x224 to fit the network: **mogrify -resize 224x224 &ast;.png**
6. Generate names.csv: **ls image-folder/ > names.csv**
7. Provide image path and names.csv in thanh_get_feature_vectors.py, then run: **python thanh_get_feature_vectors.py**
8. Provide image path and names.csv in thanh_get_similar_faces.py, then run: **python thanh_get_similar_faces.py**
9. The output html webpage is in results/ folder.
