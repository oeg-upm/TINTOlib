###########################################################
################    Constants    ##############################
###########################################################

#General
images_subfolder_name="images"
target_column_name="values"

#Problem Type
supervised="supervised"
classification="classification"
regression="regression"
unsupervised="unsupervised"
allowed_values_for_problem = [classification,supervised, unsupervised, regression]

#Ouput Data format
npy_format="npy"
png_format='png'
format_values_allowed=[png_format,npy_format]

# User Messages
generating_images_message="Generating and saving the synthetic images"
images_generated_message="Images generated and saved"
untrained_model_message="The model must be fitted before calling 'transform'. Please call 'fit' first."
invalid_input_data_message="data must be a string (file path) or a pandas DataFrame."

#Assigner
bin_assigner='bin'
bin_digitize_assigner='BinDigitize'
quantile_assigner='quantile_transform'
pixel_centroids_assigner='PixelCentroidsAssigner'
relevance_assigner='RelevanceAssigner'

#Optimizer Algorithms
linear_sum_assigner='lsa'
greedy_assigner='greedy'

#Reduction Dimensionality
pca_algorithm='PCA'
tsne_algorithm='t-SNE'
kpca_algorithm='KPCA'

#Options
avg_option="avg"
max_min_option="max/min"
zero_option="zero"
stack_option="stack"
relevance_option="rev"



