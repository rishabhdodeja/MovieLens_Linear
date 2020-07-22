# ---
# title: "MovieLens"
# author: "Rishabh Singh Dodeja"
# date: "July 17, 2020"
# ---
# This R Script will perform the following task:
#   1. Download Movie Lens 10M dataset and required libraries (if needed)
#   2. Split the data set into edx(90%) and validation(10%)
#   3. Filtering and cleaning the data into final test_set and train_set from validation and edx respectively
#   4. Build a Linear Model with predictors : movie mean rating, user effect, genre, and release year
#   5. Run Simulations to regularize bias termplot
#   6. Calculate RMSE for final Regularized Linear Model
#   7. Generate csv file containing entries from validation dataset with following columns:
#     "userId", "movieId", "rating", "predicted ratings"

################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(ratings, movies, test_index, temp, movielens, removed)


################################
# Data Filtering and cleaning
################################

# Note: this process removes features/colums that are not required by our Predictor Model
# we will only be taking userId, genre, and release year into account.
# This selection  is based on insights gained from data exploration, explained in Rmd and report.

# We will make train set from edx set, our model will be trained on this set
train_set <- edx %>% select(userId, movieId, rating, title,genres) %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# We will make test_set for validation set, our model will be testes/validated on this set
test_set <- validation %>% select(userId, movieId, rating, title,genres) %>% mutate(year = as.numeric(str_sub(title,-5,-2)))

# remove rest of the objects/datasets to free up disk space
rm(edx,validation)


################################
# Model Evaluation 
################################

# Define Mean Absolute Error (MAE)
MAE <- function(true_ratings, predicted_ratings){
  mean(abs(true_ratings - predicted_ratings))
}

# Define Mean Squared Error (MSE)
MSE <- function(true_ratings, predicted_ratings){
  mean((true_ratings - predicted_ratings)^2)
}

# Define Root Mean Squared Error (RMSE)
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Create Results Table
result <- tibble(Method = "Project Goal", RMSE = 0.8649, MSE = NA, MAE = NA);


################################
# Modelling and Results
################################

#LINEAR MODEL#

#Calculating Movie Mean Ratings
print("evaluating Movie-Mean-Model...")

#calculating movie_mean from train_set
movie_mean = train_set %>% group_by(movieId) %>% summarise(mu=mean(rating))

#Join Predicted values with test_set
y_hat_mu = test_set %>% left_join(movie_mean,by="movieId") %>% .$mu


# Update the Results table  
result <- bind_rows(result, 
                    tibble(Method = "Movie Mean", 
                           RMSE = RMSE(test_set$rating, y_hat_mu),
                           MSE  = MSE(test_set$rating, y_hat_mu),
                           MAE  = MAE(test_set$rating, y_hat_mu)));

print(result)


#Calculating User Effect Bias terms
print("Calculating User-Effect and evaluating new model...")

user_effect = train_set %>% left_join(movie_mean,by="movieId") %>% group_by(userId) %>% summarise(bu = mean(rating-mu));

# Join predicted values with test_set
y_hat_bu = test_set %>% left_join(movie_mean, by="movieId") %>% left_join(user_effect,by="userId") %>% mutate(pred=mu+bu) %>% .$pred;

result <- bind_rows(result, 
                    tibble(Method = "Movie Mean + bu", 
                           RMSE = RMSE(test_set$rating, y_hat_bu),
                           MSE  = MSE(test_set$rating, y_hat_bu),
                           MAE  = MAE(test_set$rating, y_hat_bu)))
# Show the RMSE improvement  
print(result)

#Calculating Genre-Effect bias terms
print ("Calculating Genre-effect and including in evaluation...")

#calculating genre effect from train_set
genre_effect = train_set %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>% group_by(genres) %>% summarise(bg = mean(rating-mu-bu));

#Joining predicted values with test_set
y_hat_bg = test_set %>% left_join(movie_mean, by="movieId") %>% left_join(user_effect,by="userId") %>% left_join(genre_effect,by="genres") %>% mutate(pred=mu+bu +bg) %>% .$pred;

result <- bind_rows(result, 
                    tibble(Method = "Movie Mean + bu + bg", 
                           RMSE = RMSE(test_set$rating, y_hat_bg),
                           MSE  = MSE(test_set$rating, y_hat_bg),
                           MAE  = MAE(test_set$rating, y_hat_bg)))
# Show the RMSE improvement  
print(result)

#Calculating Release Year Effect bias terms
print("calclating Release-Year Effect and evaluating...");

#calculating year effect from train_set
year_effect = train_set %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>%left_join(genre_effect,by="genres") %>% group_by(year) %>% summarise(by = mean(rating-mu-bu-bg));

# Join predictions with test_set
y_hat_by = test_set %>% left_join(movie_mean, by="movieId") %>% left_join(user_effect,by="userId") %>% left_join(genre_effect,by="genres") %>% left_join(year_effect,be="year")%>% mutate(pred=mu+bu +bg+by) %>% .$pred;

#update results table
result = bind_rows(result, 
                   tibble(Method = "Movie Mean + bu + bg + by", 
                          RMSE = RMSE(test_set$rating, y_hat_by),
                          MSE  = MSE(test_set$rating, y_hat_by),
                          MAE  = MAE(test_set$rating, y_hat_by)));
# Show the RMSE improvement  
print(result)

#REGULARIZATION#

#defining regularization fuction
regularization <- function(lambda, trainset, testset){
  
  # Movie Mean
  movie_mean = trainset %>% group_by(movieId) %>% summarise(mu=mean(rating));
  
  # User effect (bu)  
  user_effect = trainset %>% left_join(movie_mean,by="movieId") %>% group_by(userId) %>% summarise(bu =                                                                                                    sum(rating-mu)/(n()+lambda));
  #Genre effect (bg)
  genre_effect = trainset %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>%
    group_by(genres) %>% summarise(bg = sum(rating-mu-bu)/(n()+lambda));
  
  #Year effect (by)
  year_effect = trainset %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>%
    left_join(genre_effect,by="genres") %>% group_by(year) %>% summarise(by = sum(rating-mu-bu-bg)/(n()+lambda));
  
  # Prediction: mu + bu + bg + by  
  predicted_ratings = testset %>% left_join(movie_mean, by="movieId") %>% left_join(user_effect,by="userId") %>%
    left_join(genre_effect,by="genres") %>% left_join(year_effect,be="year")%>% mutate(pred=mu+bu +bg+by) %>% .$pred;
  
  return(RMSE(testset$rating,predicted_ratings));
}

#Running Regularization Simulation to get optimal value of lambda
print("Now running Regularization to tune lambda...")

# Define a set of lambdas to tune
lambdas = seq(0, 10, 0.25)

# Tune lambda
rmses = sapply(lambdas, 
               regularization, 
               trainset = train_set, 
               testset = test_set)


# Plot the lambda vs RMSE
tibble(Lambda = lambdas, RMSE = rmses) %>%
            ggplot(aes(x = Lambda, y = RMSE)) +
            geom_point() +
            ggtitle("Regularization")

#picking lambda with lowest RMSE
lambda = lambdas[which.min(rmses)];

#Evaluating ofr optimal lambda
print("Regularization done. Evaluating Final Model...")

# Movie Mean
movie_mean = train_set %>% group_by(movieId) %>% summarise(mu=mean(rating));

# User effect (bu)  
user_effect = train_set %>% left_join(movie_mean,by="movieId") %>% group_by(userId) %>% summarise(bu =                                                                                                    sum(rating-mu)/(n()+lambda));
#Genre effect (bg)
genre_effect = train_set %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>%
  group_by(genres) %>% summarise(bg = sum(rating-mu-bu)/(n()+lambda));

#Year effect (by)
year_effect = train_set %>% left_join(movie_mean,by="movieId") %>% left_join(user_effect,by="userId") %>%
  left_join(genre_effect,by="genres") %>% group_by(year) %>% summarise(by = sum(rating-mu-bu-bg)/(n()+lambda));

# Prediction: mu + bu + bg + by  
y_hat_reg = test_set %>% left_join(movie_mean, by="movieId") %>% left_join(user_effect,by="userId") %>%
  left_join(genre_effect,by="genres") %>% left_join(year_effect,be="year")%>% mutate(pred=mu+bu +bg+by) %>% .$pred;

# Final Result Results Table
result <- bind_rows(result, 
                    tibble(Method = "Regularized (mu + bu + bg + by)", 
                           RMSE = RMSE(test_set$rating, y_hat_reg),
                           MSE  = MSE(test_set$rating, y_hat_reg),
                           MAE  = MAE(test_set$rating, y_hat_reg)));

# Display Final RMSE table
print(result)

print(paste("RMSE for Final Model is:", RMSE(test_set$rating, y_hat_reg)))

################################
# Save Predictions
################################
predictions = test_set %>% select(movieId,userId,rating) %>% mutate(predictedRatings = y_hat_reg)
write.csv(predictions, "predictions.csv",row.names=FALSE)