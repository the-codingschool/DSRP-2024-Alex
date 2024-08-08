# Load necessary libraries
library(ggplot2)
library(textTinyR)
library(Matrix)  # For sparseMatrix
library(data.table)  # For data manipulation
library(proxy)   # For similarity computation
library(microbenchmark)  # For time benchmarking

# Load the data
docword_file <- "/Users/derinozkan/Desktop/bag+of+words 3/docword.nytimes.txt.gz"
docword_nyt <- fread(docword_file, header = FALSE)
setnames(docword_nyt, c("V1", "V2", "V3"), c("doc_id", "word_id", "count"))

# Function to create subsets and Document-Term Matrix (DTM)
create_dtm_subset <- function(doc_limit, word_limit) {
  docword_nyt_subset <- docword_nyt[doc_id <= doc_limit & word_id <= word_limit]
  dtm_subset <- sparseMatrix(i = docword_nyt_subset$doc_id, j = docword_nyt_subset$word_id, x = docword_nyt_subset$count,
                             dims = c(doc_limit, word_limit))
  return(dtm_subset)
}

# Function to apply TF-IDF transformation
apply_tfidf <- function(dtm) {
  # Compute term frequency (tf)
  tf <- dtm
  
  # Compute document frequency (df)
  df <- colSums(dtm > 0)
  
  # Compute inverse document frequency (idf)
  idf <- log(nrow(dtm) / df)
  
  # Apply tf-idf transformation
  tfidf <- tf %*% Diagonal(x = idf)
  
  return(tfidf)
}

# Function to compute dot product
compute_dot_product <- function(dtm) {
  normalized_dtm <- normalize_matrix(dtm)
  similarity_matrix <- tcrossprod(normalized_dtm)
  return(similarity_matrix)
}

# Function to normalize a matrix
normalize_matrix <- function(mat) {
  row_norms <- sqrt(rowSums(mat^2))
  row_norms[row_norms == 0] <- 1  # Avoid division by zero
  norm_mat <- mat / row_norms
  return(norm_mat)
}

# Function to compute SimHash
compute_simhash <- function(dtm, n_bits = 16) {
  hash_vector <- function(vec, n_bits) {
    hash_vec <- rep(0, n_bits)
    random_projections <- matrix(rnorm(n_bits * length(vec)), ncol = n_bits)
    for (i in seq_along(vec)) {
      hash_vec <- hash_vec + vec[i] * random_projections[i, ]
    }
    return(ifelse(hash_vec >= 0, 1, 0))
  }
  
  simhashes <- t(apply(dtm, 1, function(row) hash_vector(as.numeric(row), n_bits)))
  return(simhashes)
}

# Function to compute MinHash signatures
compute_minhash_signatures <- function(dtm, num_hashes = 50) {
  dtm_list <- apply(dtm, 1, function(row) {
    terms <- rep(1:ncol(dtm), times = row)
    paste(terms, collapse = " ")
  })
  names(dtm_list) <- 1:nrow(dtm)
  corpus <- TextReuseCorpus(text = dtm_list, tokenizer = tokenize_words, minhash_func = minhash_generator(num_hashes))
  minhash_signatures <- lapply(corpus, function(doc) doc$minhash)
  minhash_signatures_matrix <- do.call(rbind, minhash_signatures)
  return(minhash_signatures_matrix)
}

# Evaluate time complexity with varying dataset sizes
dataset_sizes <- list(small = list(doc_limit = 100, word_limit = 500),
                      medium = list(doc_limit = 500, word_limit = 1000),
                      large = list(doc_limit = 1000, word_limit = 2000))

time_results <- list()

for (size in names(dataset_sizes)) {
  cat("Evaluating size:", size, "\n")
  dtm_subset <- create_dtm_subset(dataset_sizes[[size]]$doc_limit, dataset_sizes[[size]]$word_limit)
  
  # Apply TF-IDF transformation
  tfidf_matrix <- apply_tfidf(dtm_subset)
  
  # Benchmark the algorithms
  results <- microbenchmark(
    tfidf = {
      apply_tfidf(dtm_subset)
    },
    dot_product = {
      compute_dot_product(tfidf_matrix)
    },
    simhash = {
      compute_simhash(dtm_subset)
    },
    minhash = {
      compute_minhash_signatures(dtm_subset)
    },
    times = 10
  )
  
  print(paste("Time taken for", size, ":"))
  print(results)
  
  time_results[[size]] <- results
}

# Optional: Print summarized results
for (size in names(time_results)) {
  print(paste("Summary of time taken for", size, ":"))
  print(summary(time_results[[size]]))
}

