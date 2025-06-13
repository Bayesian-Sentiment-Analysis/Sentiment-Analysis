library(readr)
library(dplyr)
library(stringr)
library(brms)
library(tidyr)
library(Matrix)
library(caret)

df <- read_csv("C:/Users/mdars/.vscode/cpp/SentimentAnanlysis/amazon_reviews.csv") %>%
  select(overall, reviewText) %>%
  filter(!is.na(overall), !is.na(reviewText))

preprocess <- function(text) {
  text <- tolower(text)
  text <- str_replace_all(text, "[[:punct:]]", " ")
  words <- unlist(str_split(text, "\\s+"))
  words <- words[words != ""]
  return(words)
}

processed_reviews <- lapply(df$reviewText, preprocess)
vocab <- unique(unlist(processed_reviews))

rows <- rep(seq_along(processed_reviews), sapply(processed_reviews, length))
cols <- unlist(lapply(processed_reviews, function(words) match(words, vocab)))
valid <- !is.na(cols)
rows <- rows[valid]
cols <- cols[valid]


bow_sparse <- sparseMatrix(
  i = rows,
  j = cols,
  x = 1,
  dims = c(length(processed_reviews), length(vocab)),
  dimnames = list(NULL, vocab)
)

ratings <- df$overall
bow_df <- as.data.frame(as.matrix(bow_sparse))
colnames(bow_df) <- make.names(colnames(bow_df), unique = TRUE)
model_df <- cbind(Rating = ratings, bow_df)

# ARGUMENT =====
# Keep top MAX_WORDS most frequent words
max_words <- 500
word_freq <- colSums(bow_df)
top_words <- names(sort(word_freq, decreasing = TRUE))[1:max_words]
model_df <- model_df[, c("Rating", top_words)]

# Convert Rating to ordered factor
model_df$Rating <- factor(model_df$Rating, levels = 1:5, ordered = TRUE)

# ARGUMENT =====
# Split into train and test sets
set.seed(42)
n <- nrow(model_df)
train_idx <- sample(seq_len(n), size = 0.8 * n)
train_df <- model_df[train_idx, ]
test_df  <- model_df[-train_idx, ]

# Define formula
formula <- as.formula(
  paste("Rating ~", paste(paste0("", top_words, ""), collapse = " + "))
)
print(head(formula))
# Fit ordinal regression model
fit <- tryCatch({
  cat("5a. Starting brm()...\n")
  brm(
    formula = formula,
    data = train_df,
    family = cumulative("logit"),
    chains = 2,
    iter = 1000,
    cores = 4,
    seed = 42,
    save_pars = save_pars("all")
  )
}, error = function(e) {
  cat("ERROR in brm():", e$message, "\n")
  return(NULL)
})

if (is.null(fit)) {
  cat("Model fitting failed - stopping here\n")
  stop("Cannot proceed without fitted model")
}

cat("6. Model fitted successfully!\n")
saveRDS(fit, file = "brm_iter=1000_n_words=500_n_chains=2.rds")

# Predict class probabilities
pred_probs <- posterior_epred(fit, newdata = test_df)  # dim: [samples, obs, categories]
pred_probs_mean <- apply(pred_probs, c(2, 3), mean)    # mean over samples

# Predicted class = argmax of probability
pred_classes <- apply(pred_probs_mean, 1, which.max)
true_classes <- as.integer(test_df$Rating)

# Evaluation
accuracy <- mean(pred_classes == true_classes)
cat("Test Accuracy:", round(accuracy, 3), "\n")
plot(fit$data$good)
# Confusion matrix
conf_mat <- confusionMatrix(
  factor(pred_classes, levels = 1:5),
  factor(true_classes, levels = 1:5)
)
print(conf_mat)

# Optional plots
# plot(fit)
summary(fit)
summary_df <- as.data.frame(summary(fit)$fixed)
#kf <- kfold(fit, k=5)
#pp_check(fit)
set.seed(42)
K <- 5
folds <- caret::createFolds(model_df$Rating, k = K, list = TRUE, returnTrain = FALSE)
#library(brms)
library(matrixStats)

elpd_vals <- numeric(K)

for (k in 1:K) {
  cat("Running fold", k, "\n")
  
  test_idx  <- folds[[k]]
  train_idx <- setdiff(seq_len(nrow(model_df)), test_idx)
  
  train_data <- model_df[train_idx, ]
  test_data  <- model_df[test_idx, ]
  
  fit_k <- brm(
    formula = formula,
    data = train_data,
    family = cumulative("logit"),
    chains = 2,
    iter = 1000,
    seed = 42,
    refresh = 100
  )
  cat("Model fitted successfully for fold", k, "\n")
  # Compute expected log-likelihood
  log_lik_mat <- log_lik(fit_k, newdata = test_data)
  elpd_vals[k] <- sum(log(colMeans(exp(log_lik_mat))))  # log pointwise predictive density
}
print(elpd_vals)
tot = 0
# Check if the issue is in individual folds
tot <- sum(elpd_vals, na.rm = TRUE)  # This removes NaN values automatically
deviance <- -2 * tot
cat("Total ELPD:", round(tot, 3), "\n")
cat("Deviance:", round(deviance, 3), "\n")

deviance = -2*tot
cat("Deviance is", round(deviance,3))
