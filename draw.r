#1. Read and output 2022-2050 prediction results CSV -------------------------------------------------------
library(tidyverse)

# Set main directory path (modify according to your actual path)
main_dir <- "Model"

# Get all model directories
model_dirs <- list.dirs(main_dir, full.names = TRUE, recursive = FALSE)

# Initialize empty tibble to store results
combined_data <- tibble()

# Process each model directory
for (model_dir in model_dirs) {
  # Extract model name from directory path
  model_name <- basename(model_dir)
  
  # Construct results directory path
  result_dir <- file.path(model_dir, "2022-2050")
  
  # Get all CSV files in results directory
  csv_files <- list.files(result_dir, pattern = "\\.csv$", full.names = TRUE)
  
  # Process each CSV file
  for (csv_file in csv_files) {
    # Clean age group name from filename
    age_group <- basename(csv_file) %>% 
      tools::file_path_sans_ext() %>% 
      if_else(. == "result_less_5_years", "0-4_years", .) %>% 
      gsub("^result_", "", .)
    
    tryCatch({
      # Read and process data
      df <- read_csv(
        csv_file,
        col_names = c("year", "predict"),
        col_types = cols(
          year = col_integer(),
          predict = col_double()
        ),
        na = c("", "NA", "NULL")
      ) %>% 
        # Remove columns with all NAs
        select(where(~!all(is.na(.)))) %>%
        # Remove rows with NA in year or predict
        drop_na(year, predict) %>%  # New filtering logic
        # Add metadata
        mutate(
          model = model_name,
          age = age_group,
          .before = everything()
        )
      
      # Combine data
      combined_data <- bind_rows(combined_data, df)
      
    }, error = function(e) {
      message("Error processing file: ", csv_file)
      message("Error message: ", e$message)
    })
  }
}

# Verify final data
glimpse(combined_data)

# Optional: Save results
write_csv(combined_data, "2022-2050 predictions.csv")




#2. Read and output 2011-2021 test set results -------------------------------------------------------
library(tidyverse)

# Set main directory path (modify according to actual situation)
main_dir <- "Model"

# Get all model directories
model_dirs <- list.dirs(main_dir, full.names = TRUE, recursive = FALSE)

# Initialize empty dataframe
combined_data <- tibble()

# Process each model directory
for (model_dir in model_dirs) {
  # Extract model name
  model_name <- basename(model_dir)
  
  # Construct age results directory path (Modification 1: Path correction)
  age_result_dir <- file.path(model_dir, "2011-2021")
  
  # Get all CSV files (Modification 2: Add existence check)
  if(dir.exists(age_result_dir)){
    csv_files <- list.files(age_result_dir, pattern = "\\.csv$", full.names = TRUE)
    
    # Process each CSV file
    for (csv_file in csv_files) {
      # Extract age group name (Modification 3: Simplified processing logic)
      age_group <- tools::file_path_sans_ext(basename(csv_file))
      
      tryCatch({
        # Read data (Modification 4: Add actual column processing)
        df <- read_csv(
          csv_file,
          col_names = c("year", "actual", "predict"), # Three columns correspond
          col_types = cols(
            year = col_integer(),
            actual = col_double(),
            predict = col_double()
          ),
          na = c("", "NA", "NULL")
        ) %>% 
          # Data cleaning (Modification 5: Add actual column validation)
          select(where(~!all(is.na(.)))) %>%
          drop_na(year, actual, predict) %>% # Three-column validation
          mutate(
            model = model_name,
            age = age_group,
            .before = 1 # Ensure correct column order
          )
        
        combined_data <- bind_rows(combined_data, df)
        
      }, error = function(e) {
        message("File processing error: ", csv_file)
        message("error message: ", e$message)
      })
    }
  } else {
    warning(paste("directory does not exist:", age_result_dir))
  }
}

# Verify data structure (Modification 6: Add column order validation)
required_columns <- c("model", "age", "year", "actual", "predict")
if(all(required_columns %in% names(combined_data))){
  cat("Data structure validation passed:\n")
  glimpse(combined_data)
} else {
  stop("Missing columns: ", 
       paste(setdiff(required_columns, names(combined_data)), collapse = ", "))
}

# Save results (optional)
write_csv(combined_data, "2011-2021 test.csv")




#3. Calculate test set key metrics (RMSE, MAE, MAPE, R2) -----------------------------------------
library(dplyr)
library(readr)

# Read test data
# Assume data file is in current working directory with columns:
# model: Model name
# age: Age group
# year: Year (2011-2021)
# actual: Actual values
# predict: Predicted values
test_data <- read_csv("2011-2021 test.csv", 
                      col_types = cols(
                        model = col_character(),
                        age = col_character(),
                        year = col_integer(),
                        actual = col_double(),
                        predict = col_double()
                      ))

# Define metric calculation function
calculate_metrics <- function(actual, predict) {
  # Remove observations with NA
  complete_idx <- complete.cases(actual, predict)
  actual_clean <- actual[complete_idx]
  predict_clean <- predict[complete_idx]
  
  # Calculate basic metrics
  errors <- actual_clean - predict_clean
  n <- length(actual_clean)
  
  # RMSE (Root Mean Square Error)
  rmse <- sqrt(mean(errors^2))
  
  # MAE (Mean Absolute Error)
  mae <- mean(abs(errors))
  
  # MAPE (Mean Absolute Percentage Error)
  # Handle cases where actual value is 0
  non_zero_idx <- actual_clean != 0
  if (sum(non_zero_idx) > 0) {
    mape <- 100 * mean(abs(errors[non_zero_idx] / actual_clean[non_zero_idx]))
  } else {
    mape <- NA_real_
  }
  
  # R² (Coefficient of Determination)
  if (n > 1) {
    ss_total <- sum((actual_clean - mean(actual_clean))^2)
    ss_residual <- sum(errors^2)
    r2 <- 1 - (ss_residual / ss_total)
  } else {
    r2 <- NA_real_
  }
  
  return(data.frame(RMSE = rmse, MAE = mae, MAPE = mape, R2 = r2))
}

# Calculate metrics for each model-age group
metrics_df <- test_data %>%
  group_by(model, age) %>%
  summarise(
    calculate_metrics(actual, predict),
    .groups = "drop"
  )

# Calculate rankings within each age group
ranked_df <- metrics_df %>%
  group_by(age) %>%  # Group by age group for ranking
  mutate(
    # RMSE ranking (smaller is better, ascending order)
    RMSE_rank = rank(RMSE, ties.method = "min"),
    # MAE ranking (smaller is better, ascending order)
    MAE_rank = rank(MAE, ties.method = "min"),
    # MAPE ranking (smaller is better, handle NA)
    MAPE_rank = rank(MAPE, ties.method = "min", na.last = "keep"),
    # R² ranking (larger is better, descending order)
    R2_rank = rank(-R2, ties.method = "min", na.last = "keep")
  ) %>%
  ungroup()

# Calculate average ranking and format output
final_result <- ranked_df %>%
  rowwise() %>%  # Calculate average rank row-wise
  mutate(
    Avg.Rank = mean(c(RMSE_rank, MAE_rank, MAPE_rank, R2_rank), na.rm = TRUE)
  ) %>%
  ungroup() %>%
  # Adjust column order
  select(
    model, age,
    RMSE, RMSE_rank,
    MAE, MAE_rank,
    MAPE, MAPE_rank,
    R2, R2_rank,
    Avg.Rank
  ) %>%
  # Sort by average rank (optional)
  arrange(age, Avg.Rank)

# View results
print(final_result, n = 10)

# Save results to CSV file (optional)
write_csv(final_result, "model_performance_ranking.csv")




#4. Below is plotting code, not analysis code -----------------------------------
library(tidyverse)
library(patchwork)

# Read and sort data -----------------------------------------------------------
df <- read_csv("0-4model.csv") %>% 
  mutate(model = fct_reorder(model, Avg.Rank) %>% fct_rev())

# Custom color scheme (with transparency) ---------------------------------------
metric_colors <- list(
  RMSE = list(
    colors = c("#FFE4E180", "#e38d8c80", "#dd181d80"), # Dark red -> Light red (alpha=0.5)
    limits = c(0, 0.5)
  ),
  MAE = list(
    colors = c("#FFEC8B80", "#f2b74d80", "#FFA50080"), # Dark orange -> Light yellow (alpha=0.5)
    limits = c(0, 1)
  ),
  MAPE = list(
    colors = c("#FFFACD80", "#FFD70080", "#8B750080"), # Dark gold -> Light gold (alpha=0.5)
    limits = c(0, 10)
  ),
  R2 = list(
    colors = c("#5f86c480", "#00BFFF80", "#E0FFFF80"), # Blue gradient (alpha=0.5)
    limits = c(-1, 1)
  )
)

# Bar chart color configuration -------------------------------------------------
bar_colors <- c(
  RMSE_rank = "#FFB6C1",  # Light pink
  MAE_rank = "#FFDAB9",   # Peach
  MAPE_rank = "#EEE8AA",  # Pale gold
  R2_rank = "#87CEFA",    # Sky blue
  Avg.Rank = "#D8BFD8"    # Thistle
)
bar_border <- "#b4b6b6"  # Brown border

# Heatmap generation function ---------------------------------------------------
create_heatmap <- function(metric) {
  ggplot(df, aes(x = 1, y = model)) +
    geom_tile(
      aes(fill = .data[[metric]]),
      color = bar_border,
      linewidth = 0.8,
      width = 0.85,
      height = 0.75
    ) +
    geom_text(
      aes(label = round(.data[[metric]], 2)),
      size = 3.5,
      color = "black"
    ) +
    scale_fill_gradientn(
      colors = metric_colors[[metric]]$colors,
      limits = metric_colors[[metric]]$limits,
      na.value = "gray90",
      oob = scales::squish
    ) +
    labs(title = metric) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
      plot.margin = margin(2, 2, 2, 2, "mm")
    )
}

# Bar chart generation function (Adjusted grid line order) ----------------------
create_barplot <- function(rank_metric) {
  ggplot(df, aes(x = .data[[rank_metric]], y = model)) +
    # Draw background grid lines first
    geom_vline(
      xintercept = c(0, 5, 10),
      color = "gray80",
      linewidth = 0.3
    ) +
    # Draw bar chart on top of grid lines
    geom_col(
      fill = bar_colors[rank_metric],
      color = bar_border,
      linewidth = 0.8,
      width = 0.65
    ) +
    scale_x_continuous(
      limits = c(0, 10),
      breaks = c(0, 5, 10),
      position = "top"
    ) +
    labs(title = str_remove(rank_metric, "_rank")) +
    theme_minimal() +
    theme(
      axis.text.y = element_blank(),
      panel.grid = element_blank(),
      plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
      axis.text.x = element_text(size = 8, color = "black"),
      plot.margin = margin(2, 2, 2, 2, "mm"),
      panel.background = element_rect(fill = NA, color = "gray80", linewidth = 0.5),
      panel.border = element_rect(fill = NA, color = "gray80", linewidth = 0.5)
    )
}

# Generate plot components ------------------------------------------------------
heatmaps <- map(c("RMSE", "MAE", "MAPE", "R2"), create_heatmap)
bars <- map(names(bar_colors)[1:4], create_barplot)

# Average rank bar chart
avg_rank_bar <- create_barplot("Avg.Rank") + 
  labs(title = "Avg.Rank") +
  scale_fill_manual(values = bar_colors["Avg.Rank"])

# Combine final plot ------------------------------------------------------------
final_plot <- wrap_plots(
  heatmaps[[1]], bars[[1]],
  heatmaps[[2]], bars[[2]],
  heatmaps[[3]], bars[[3]],
  heatmaps[[4]], bars[[4]],
  avg_rank_bar,
  ncol = 9,
  widths = c(0.7, 1.1, 0.7, 1.1, 0.7, 1.1, 0.7, 1.1, 1.3)
) + 
  plot_annotation(
    title = "Model Performance Dashboard",
    theme = theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      plot.background = element_rect(fill = "white", color = "gray80", linewidth = 1)
    )
  )

# Display plot
print(final_plot)

# Save plot (adjust dimensions for readability)
ggsave("model_comparison.png", final_plot, 
       width = 16, height = 6, dpi = 300)

# Second plot -------------------------------------------------------------------
library(tidyverse)
library(ggthemes)

# Read data
df <- read.csv("deathRate.csv")

# Convert to long format
df_long <- df %>% 
  pivot_longer(
    cols = -year,
    names_to = "Metric",
    values_to = "Rate"
  )

# Draw optimized point-line plot
ggplot(df_long, aes(x = year, y = Rate, color = Metric)) +
  # Smooth curve (using spline interpolation)
  geom_smooth(
    aes(group = Metric),
    method = "lm",
    formula = y ~ splines::bs(x, 15),  # 15-node spline interpolation
    se = FALSE,
    size = 1.5,         # Thicker line
    alpha = 0.7,
    lineend = "round",  # Rounded line ends
    linejoin = "round"  # Rounded line joins
  ) +
  # Original data points
  geom_point(
    size = 2,           # Slightly larger point size
    shape = 19,
    alpha = 0.8,
    stroke = 0.5        # Point border thickness
  ) +
  # 2021 reference line
  geom_vline(
    xintercept = 2021,
    linetype = "longdash",
    color = "#2F4F4F",  # Dark slate gray
    linewidth = 0.6,
    alpha = 0.7
  ) +
  # Add year annotation
  annotate(
    "text",
    x = 2021, 
    y = max(df_long$Rate) * 0.95,
    label = "2021",
    angle = 90,
    vjust = -0.5,
    color = "#2F4F4F",
    size = 3.5
  ) +
  scale_color_manual(
    values = c("#f2b74d", "#e89b7e"),  # More vivid scientific colors
    labels = c("All Ages", "Age-Std")
  ) +
  scale_x_continuous(
    breaks = seq(1980, 2050, by = 10),
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    limits = c(floor(min(df_long$Rate)*0.95), ceiling(max(df_long$Rate)*1.05)),
    n.breaks = 8
  ) +
  labs(
    title = "Disease Rate Trends (1980-2050)",
    subtitle = "With 2021 Reference Marker",
    x = "Year",
    y = "Rate per 100,000 Population",
    color = "Metric Type"
  ) +
  theme_foundation() +
  theme(
    plot.title = element_text(
      size = 16,
      face = "bold",
      hjust = 0.5,
      color = "#2F4F4F"
    ),
    plot.subtitle = element_text(
      size = 12,
      hjust = 0.5,
      margin = margin(b = 15)
    ),
    axis.title = element_text(
      size = 12,
      color = "#4A4A4A"
    ),
    axis.text = element_text(
      size = 10, 
      color = "#6B6B6B"
    ),
    legend.position = c(0.12, 0.88),
    legend.background = element_rect(
      fill = alpha("white", 0.9),
      color = "#D3D3D3"
    ),
    legend.key = element_blank(),
    panel.grid.major = element_line(
      color = "#EAEAEA",
      linewidth = 0.3
    ),
    panel.background = element_rect(fill = "#F9F9F9"),
    plot.background = element_rect(fill = "white")
  )
