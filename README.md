**Data Collection:**

This project uses the data scraped from imdb website by https://github.com/sundeepblue/movie_rating_prediction/ . 

The Source is  a CSV file 'movie_metadata.csv' (1.5MB) that has the below features.

"movie_title"
"color"
"num_critic_for_reviews"
"movie_facebook_likes" "duration"
"director_name"
"director_facebook_likes"
"actor_3_name" "actor_3_facebook_likes"
"actor_2_name"
"actor_2_facebook_likes"
"actor_1_name" "actor_1_facebook_likes"
"gross"
"genres"
"num_voted_users"
"cast_total_facebook_likes" "facenumber_in_poster"
"plot_keywords"
"movie_imdb_link"
"num_user_for_reviews"
"language"
"country"
"content_rating"
"budget"
"title_year"
"imdb_score"
"aspect_ratio"

**Methodology**

This project uses three Regressors - KNN, Random Forest Model and Gradient Boosting in Python from the sklearn package, uses techniques like One Hot Encoding, 5 Fold Cross Validation, Grid Search CV for Hyperparameter tuning to identify the best model.

**Evaluation**

RMSE (Root Mean Square Error)  and R^2 were used as the Evaluation metrics for the regressors


**Contact**
Please contact for yamineesh@gmail.com for any questions regarding the project
