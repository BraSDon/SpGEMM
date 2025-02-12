library(tidyverse)
library(scales)
library(jsonlite)

data <- as_tibble(fromJSON(file("benchmark.json"))$benchmarks) %>%
  transmute(name = name, time = real_time / 1e6) %>%
  extract(
    name,
    into = c("name", "type", "size", "sparsity"),
    regex = "BM_([a-zA-Z_]+)<([a-zA-Z]+), ([0-9]+)>/([0-9]+)",
    remove = FALSE
  ) %>%
  mutate(
    name = fct_relabel(factor(name), ~ gsub("^BM_", "", .)), 
    size = as.numeric(size),
    sparsity = as.numeric(sparsity)
  )
pdf("plots.pdf", width = 10, height = 6)



# -+--+--+--+--+--+--+--+--+--+--+--+- Int -+--+--+--+--+--+--+--+--+--+--+--+- #
int_data <- data %>% filter(type == "int")
int_plot <- ggplot(int_data, aes(x = log2(size), y = time, color = name)) +
  geom_line(linewidth = 0.8) +
  geom_point(aes(shape = factor(sparsity)), size = 2) +
  scale_x_continuous("Matrix Dimension (log2)", labels = math_format(2^.x), breaks = 0:20) +
  scale_y_log10("Execution Time (ms)", labels = scales::comma) +
  facet_wrap(~ sparsity, labeller = as_labeller(function(x) paste("Sparsity:", x, "%"))) +
  labs(
    title = "Execution Time on random square matrices of varying sparsity",
    color = "Implementation",
    shape = "Sparsity (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(int_plot)

# -+--+--+--+--+--+--+--+--+--+--+--+- Scaling -+--+--+--+--+--+--+--+--+--+--+--+- #
scaling_data <- as_tibble(fromJSON(file("benchmark.json"))$benchmarks) %>%
  transmute(name = name, time = real_time / 1e6) %>%
  extract(
    name,
    into = c("name", "type", "size", "threads", "sparsity"),
    regex = "BM_([a-zA-Z_]+)<([a-zA-Z]+), ([0-9]+), ([0-9]+)>/([0-9]+)",
    remove = FALSE
  ) %>%
  mutate(
    name = fct_relabel(factor(name), ~ gsub("^BM_", "", .)), 
    size = as.numeric(size),
    threads = as.numeric(threads),
    sparsity = as.numeric(sparsity)
  )

reference_time <- scaling_data %>%
  filter(threads == 1) %>%
  select(size, sparsity, runtime_1_thread = time)

scaling_data <- scaling_data %>%
  inner_join(reference_time, by = c("size", "sparsity")) %>%
  mutate(scaling_efficiency = runtime_1_thread / (threads * time)) %>%
  filter(!is.na(scaling_efficiency))

scaling_plot <- ggplot(scaling_data, aes(
    x = log2(threads),
    y = scaling_efficiency,
    color = factor(sparsity),
    shape = factor(sparsity)
  )) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  geom_vline(xintercept = 3, linetype = "dashed", color = "red", linewidth = 1) +
  scale_x_continuous("Threads (log2)", labels = math_format(2^.x), breaks = 0:10) +
  scale_y_continuous("Scaling Efficiency", labels = scales::comma) +
  labs(
    title = "Relative Scaling Efficiency",
    subtitle = "Using random square matrix of dimension 2048",
    x = "Threads (log2)",
    y = "Scaling Efficiency",
    color = "Sparsity (%)",
    shape = "Sparsity (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(scaling_plot)

# -+--+--+--+--+--+--+--+--+--+--+--+- Speedup -+--+--+--+--+--+--+--+--+--+--+--+- #
speedup_data <- int_data %>%
  filter (
    name %in% c("Gustavson_random", "Parallel_random"),
  ) %>%
  tidyr::pivot_wider(names_from = name, values_from = time) %>%
  mutate(speedup = Gustavson_random / Parallel_random)

speedup_plot <- ggplot(speedup_data, aes(
    x = log2(size), 
    y = speedup, 
    color = factor(sparsity),
    shape = factor(sparsity)
  )) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 2) +
  scale_x_continuous(
    "Matrix Dimension (log2)",
    labels = scales::math_format(2^.x),
    breaks = 0:20
  ) +
  scale_y_continuous(
    "Absolute Speedup (Gustavson / Parallel)",
    labels = scales::comma
  ) +
  labs(
    title = "Speedup of Parallel over Sequenial Gustavson. 8 Threads",
    color = "Sparsity (%)",
    shape = "Sparsity (%)"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")
print(speedup_plot)

# -+--+--+--+--+--+--+--+--+--+--+--+- Double -+--+--+--+--+--+--+--+--+--+--+--+- #
# double_data <- data %>% filter(type == "double")
# double_plot <- ggplot(double_data, aes(x = log2(size), y = time, color = name)) +
#   geom_line(size = 0.8) +
#   geom_point(aes(shape = factor(sparsity)), size = 2) +
#   scale_x_continuous("Matrix Size (log2)", labels = math_format(2^.x), breaks = -100:100*2) +
#   scale_y_log10("Execution Time (ms)", labels = scales::comma) +
#   facet_wrap(~ sparsity, labeller = as_labeller(function(x) paste("Sparsity:", x, "%"))) +
#   labs(
#     title = "Comparison of Double-based SpGEMM Benchmarks",
#     subtitle = "Execution Time vs. Matrix Size",
#     color = "Implementation",
#     shape = "Sparsity (%)"
#   ) +
#   theme_minimal() +
#   theme(legend.position = "bottom")
# print(double_plot)

# -+--+--+--+--+--+--+--+--+--+--+--+- Florida Matrices -+--+--+--+--+--+--+--+--+--+--+--+- #
data <- as_tibble(fromJSON(file("benchmark.json"))$benchmarks) %>%
  transmute(
    name = name, 
    time = real_time / 1e6  # Convert ns to ms
  )

# Extract the benchmark name and statistic (mean or stddev)
clean_data <- data %>%
  extract(
    name,
    into = c("benchmark", "statistic"),
    regex = "(BM_[^/]+)(?:/repeats:[0-9]+)?_(mean|stddev)",
    remove = FALSE
  ) %>%
  filter(statistic %in% c("mean", "stddev")) %>%
  pivot_wider(names_from = statistic, values_from = time, values_fill = list(mean = NA, stddev = NA))

expensive_bar <- clean_data %>%
  filter(benchmark %in% c(
    "BM_Gustavson_rgg_n_2_22_s0", "BM_Parallel_rgg_n_2_22_s0", "BM_Eigen_rgg_n_2_22_s0",
    "BM_Gustavson_C71", "BM_Parallel_C71", "BM_Eigen_C71",
    "BM_Gustavson_rajat31", "BM_Parallel_rajat31", "BM_Eigen_rajat31",
    "BM_Gustavson_preferentialAttachment", "BM_Parallel_preferentialAttachment", "BM_Eigen_preferentialAttachment",
    "BM_Gustavson_consph", "BM_Parallel_consph", "BM_Eigen_consph",
    "BM_Gustavson_M6", "BM_Parallel_M6", "BM_Eigen_M6"
  )) %>%
  mutate(
    matrix = case_when(
      str_detect(benchmark, "rgg_n_2_22_s0") ~ "rgg_n_2_22_s0",
      str_detect(benchmark, "rajat31") ~ "rajat31",
      str_detect(benchmark, "M6") ~ "M6",
      str_detect(benchmark, "C71") ~ "C71",
      str_detect(benchmark, "preferentialAttachment") ~ "preferentialAttachment",
      str_detect(benchmark, "consph") ~ "consph",
      TRUE ~ NA_character_
    ),
    implementation = case_when(
      str_detect(benchmark, "Gustavson") ~ "Gustavson",
      str_detect(benchmark, "Parallel") ~ "Parallel",
      str_detect(benchmark, "Eigen") ~ "Eigen",
      TRUE ~ NA_character_
    )
  ) %>%
  arrange(desc(mean))

# Separate data based on benchmark types
data_bar <- clean_data %>%
  filter(benchmark %in% c(
    "BM_Gustavson_Nemeth", "BM_Parallel_Nemeth", "BM_Eigen_Nemeth", 
    "BM_Gustavson_Lhr71c", "BM_Parallel_Lhr71c", "BM_Eigen_Lhr71c",
    "BM_Gustavson_ASIC_680ks", "BM_Parallel_ASIC_680ks", "BM_Eigen_ASIC_680ks"
  )) %>%
  mutate(
    matrix = case_when(
      str_detect(benchmark, "Nemeth") ~ "Nemeth",
      str_detect(benchmark, "Lhr71c") ~ "Lhr71c",
      str_detect(benchmark, "ASIC_680ks") ~ "ASIC_680ks",
      TRUE ~ NA_character_
    ),
    implementation = case_when(
      str_detect(benchmark, "Gustavson") ~ "Gustavson",
      str_detect(benchmark, "Parallel") ~ "Parallel",
      str_detect(benchmark, "Eigen") ~ "Eigen",
      TRUE ~ NA_character_
    )
  ) %>%
  arrange(desc(mean))

if (nrow(data_bar) > 0) {
  data_bar$matrix <- factor(data_bar$matrix, levels = unique(data_bar$matrix))
  florida_plot <- ggplot(data_bar, aes(x = matrix, y = mean, fill = implementation)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    geom_text(aes(label = ceiling(mean)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, 
              size = 3) +
    scale_y_continuous("Execution Time (ms)", labels = scales::comma) +
    labs(
      title = "Execution Time Comparison of Florida Sparse Matrices",
      x = "Matrix",
      fill = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  print(florida_plot)
}

if (nrow(expensive_bar) > 0) {
  expensive_bar$matrix <- factor(expensive_bar$matrix, levels = unique(expensive_bar$matrix))
  expensive_florida_plot <- ggplot(expensive_bar, aes(x = matrix, y = mean, fill = implementation)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
    geom_text(aes(label = ceiling(mean)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, 
              size = 3) +
    scale_y_continuous("Execution Time (ms)", labels = scales::comma) +
    labs(
      title = "Execution Time Comparison of Florida Sparse Matrices",
      x = "Matrix",
      fill = "Implementation"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  print(expensive_florida_plot)
}

# Close the PDF device
dev.off()
