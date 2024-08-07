# Load necessary libraries
library(ggplot2)
library(textTinyR)
library(textreuse)
library(Matrix)  # For sparseMatrix
library(proxy)   # For similarity computation
library(microbenchmark)
library(pryr)    # For memory profiling
library(data.table)  # For data manipulation
library(reshape2)    # For data reshaping
library(gridExtra)   # For arranging plots side by side

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
  return(tcrossprod(dtm))
}

# Function to compute SimHash
compute_simhash <- function(dtm, n_bits = 16) {
  hash_vector <- function(vec, n_bits) {
    # Initialize the hash vector
    hash_vec <- rep(0, n_bits)
    # Generate random projection vectors
    random_projections <- matrix(rnorm(n_bits * length(vec)), ncol = n_bits)
    for (i in seq_along(vec)) {
      hash_vec <- hash_vec + vec[i] * random_projections[i, ]
    }
    return(ifelse(hash_vec >= 0, 1, 0))
  }
  
  simhashes <- t(apply(dtm, 1, function(row) hash_vector(as.numeric(row), n_bits)))
  return(simhashes)
}

# Function to compute similarity based on SimHash
compute_simhash_similarity <- function(simhashes) {
  num_docs <- nrow(simhashes)
  simhash_sim <- matrix(0, nrow = num_docs, ncol = num_docs)
  
  for (i in 1:num_docs) {
    for (j in 1:num_docs) {
      simhash_sim[i, j] <- sum(simhashes[i, ] == simhashes[j, ]) / ncol(simhashes)
    }
  }
  
  return(simhash_sim)
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

# Function to compute MinHash similarity
compute_minhash_similarity <- function(signatures) {
  num_docs <- nrow(signatures)
  jaccard_sim <- matrix(0, nrow = num_docs, ncol = num_docs)
  for (i in 1:num_docs) {
    for (j in 1:num_docs) {
      jaccard_sim[i, j] <- sum(signatures[i, ] == signatures[j, ]) / ncol(signatures)
    }
  }
  return(jaccard_sim)
}

# Function to cluster documents
cluster_documents <- function(similarity_matrix, method = "average") {
  dist_matrix <- as.dist(1 - similarity_matrix)
  hc <- hclust(dist_matrix, method = method)
  return(hc)
}

# Evaluate robustness with varying dataset sizes
dataset_sizes <- list(small = list(doc_limit = 100, word_limit = 500),
                      medium = list(doc_limit = 500, word_limit = 1000),
                      large = list(doc_limit = 1000, word_limit = 2000))

robustness_results <- list()

for (size in names(dataset_sizes)) {
  cat("Evaluating size:", size, "\n")
  dtm_subset <- create_dtm_subset(dataset_sizes[[size]]$doc_limit, dataset_sizes[[size]]$word_limit)
  
  time_results <- microbenchmark(
    tfidf = {
      tfidf_matrix <- apply_tfidf(dtm_subset)
      tfidf_memory <- object_size(tfidf_matrix)
    },
    dot_product = {
      dot_product_result <- compute_dot_product(tfidf_matrix)
      dot_product_memory <- object_size(dot_product_result)
    },
    simhash_similarity = {
      simhashes <- compute_simhash(tfidf_matrix)
      simhash_similarity_result <- compute_simhash_similarity(simhashes)
      simhash_memory <- object_size(simhash_similarity_result)
    },
    minhash_similarity = {
      minhash_signatures <- compute_minhash_signatures(tfidf_matrix)
      minhash_similarity_result <- compute_minhash_similarity(minhash_signatures)
      minhash_memory <- object_size(minhash_similarity_result)
    },
    times = 10
  )
  
  print(paste("TF-IDF memory for", size, ":", tfidf_memory))
  print(paste("Dot product memory for", size, ":", dot_product_memory))
  print(paste("SimHash memory for", size, ":", simhash_memory))
  print(paste("MinHash memory for", size, ":", minhash_memory))
  
  memory_data <- data.frame(
    method = c("tfidf", "dot_product", "simhash_similarity", "minhash_similarity"),
    memory = c(as.numeric(tfidf_memory), as.numeric(dot_product_memory), as.numeric(simhash_memory), as.numeric(minhash_memory))
  )
  
  robustness_results[[size]] <- list(time = time_results, memory = memory_data)
}

# Collecting time and memory data for plotting
time_data_list <- lapply(names(robustness_results), function(size) {
  time_data <- as.data.frame(robustness_results[[size]]$time)
  time_data$size <- size
  return(time_data)
})
time_data <- do.call(rbind, time_data_list)

memory_data_list <- lapply(names(robustness_results), function(size) {
  memory_data <- robustness_results[[size]]$memory
  memory_data$size <- size
  return(memory_data)
})
memory_data <- do.call(rbind, memory_data_list)

# Melting data for ggplot
time_data_melt <- melt(time_data, id.vars = c("expr", "size"), measure.vars = "time")
memory_data_melt <- melt(memory_data, id.vars = c("method", "size"), measure.vars = "memory")

# Plotting time comparison
time_plot <- ggplot(time_data_melt, aes(x = size, y = value / 1e6, fill = expr)) +
  geom_boxplot() +
  facet_wrap(~expr, scales = "free_y") +
  labs(title = "Time Complexity Comparison Across Dataset Sizes", x = "Dataset Size", y = "Time (milliseconds)") +
  theme_minimal()

# Plotting memory comparison with logarithmic scale
memory_plot <- ggplot(memory_data_melt, aes(x = size, y = log10(value / 1e6), fill = method)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Space Complexity Comparison Across Dataset Sizes", x = "Dataset Size", y = "Log10 Memory (megabytes)") +
  theme_minimal()

# Arrange plots side by side
grid.arrange(time_plot, memory_plot, ncol = 2)

