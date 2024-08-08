# Load necessary libraries
library(Matrix)
library(data.table)
library(ggplot2)
library(pryr)  # For memory measurement

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

# Function to compute SimHash manually
compute_simhash <- function(dtm) {
  simhash_matrix <- as.matrix(dtm) %*% matrix(runif(ncol(dtm)), ncol = 1)
  simhash_matrix <- sign(simhash_matrix)
  return(simhash_matrix)
}

# Custom implementation of MinHash
compute_minhash <- function(dtm, num_hashes = 200) {
  num_docs <- nrow(dtm)
  num_terms <- ncol(dtm)
  minhash_matrix <- matrix(Inf, nrow = num_docs, ncol = num_hashes)
  
  for (i in 1:num_hashes) {
    perm <- sample(1:num_terms)
    for (j in 1:num_docs) {
      terms <- which(dtm[j, ] > 0)
      minhash_matrix[j, i] <- min(perm[terms])
    }
  }
  
  return(minhash_matrix)
}

# Function to apply TF-IDF transformation
apply_tfidf <- function(dtm) {
  tf <- dtm
  df <- colSums(dtm > 0)
  idf <- log(nrow(dtm) / df)
  tfidf <- tf %*% Diagonal(x = idf)
  return(tfidf)
}

# Function to compute Dot Product similarity
compute_dot_product <- function(dtm) {
  similarity_matrix <- tcrossprod(as.matrix(dtm))
  return(similarity_matrix)
}

# Function to measure memory usage
measure_memory <- function(expr) {
  result <- eval(expr)
  mem_usage <- object_size(result)
  return(list(memory = mem_usage, result = result))
}

# Evaluate the algorithms
dataset_sizes <- list(small = list(doc_limit = 100, word_limit = 500),
                      medium = list(doc_limit = 500, word_limit = 1000),
                      large = list(doc_limit = 1000, word_limit = 2000))

memory_results <- list()

for (size in names(dataset_sizes)) {
  cat("Evaluating size:", size, "\n")
  
  # Create DTM
  dtm_subset <- create_dtm_subset(dataset_sizes[[size]]$doc_limit, dataset_sizes[[size]]$word_limit)
  
  # Measure memory for SimHash
  simhash_memory <- measure_memory(expression(compute_simhash(dtm_subset)))
  
  # Measure memory for MinHash
  minhash_memory <- measure_memory(expression(compute_minhash(dtm_subset)))
  
  # Measure memory for TF-IDF
  tfidf_memory <- measure_memory(expression(apply_tfidf(dtm_subset)))
  
  # Measure memory for Dot Product
  dot_product_memory <- measure_memory(expression(compute_dot_product(dtm_subset)))
  
  # Record results
  memory_results[[size]] <- list(
    simhash = simhash_memory$memory,
    minhash = minhash_memory$memory,
    tfidf = tfidf_memory$memory,
    dot_product = dot_product_memory$memory
  )
}

# Print the memory usage
for (size in names(memory_results)) {
  cat("Memory usage for", size, "dataset:\n")
  cat("  SimHash: ", format(memory_results[[size]]$simhash, units = "MB"), "\n")
  cat("  MinHash: ", format(memory_results[[size]]$minhash, units = "MB"), "\n")
  cat("  TF-IDF: ", format(memory_results[[size]]$tfidf, units = "MB"), "\n")
  cat("  Dot Product: ", format(memory_results[[size]]$dot_product, units = "MB"), "\n")
}

# Prepare data for plotting
plot_data <- data.frame(
  Size = rep(names(dataset_sizes), each = 4),
  Algorithm = rep(c("SimHash", "MinHash", "TF-IDF", "Dot Product"), times = length(dataset_sizes)),
  Memory = unlist(lapply(memory_results, function(x) c(x$simhash, x$minhash, x$tfidf, x$dot_product)))
)

# Plotting Space Complexity
ggplot(plot_data, aes(x = Size, y = Memory, fill = Algorithm)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Space Complexity of Different Algorithms",
       x = "Dataset Size", y = "Memory Usage (MB)") +
  theme_minimal()

