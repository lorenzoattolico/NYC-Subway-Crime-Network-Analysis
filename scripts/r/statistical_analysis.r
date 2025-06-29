# NYC Subway Crime Network Analysis
# Statistical Analysis using Generalized Additive Models (GAMs)

# Load required libraries
library(tidyverse)  # For data manipulation
library(mgcv)       # For GAM models
library(MASS)       # For negative binomial models
library(car)        # For diagnostic functions
library(viridis)    # For color palettes
library(gridExtra)  # For arranging multiple plots

# --------------------------------------------------------
# 1. DATA PREPARATION
# --------------------------------------------------------
run_statistical_analysis <- function(input_file = "data/nyc_subway_crime_poverty_analysis.csv", 
                                     output_dir = "results") {
  
  # Create output directory if it doesn't exist
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Log function for tracking progress
  log_message <- function(msg) {
    cat(paste0(Sys.time(), " - ", msg, "\n"))
  }
  
  log_message("Starting statistical analysis")
  log_message(paste("Loading data from", input_file))
  
  # Load data
  adata <- read.csv(input_file, stringsAsFactors = FALSE)
  
  # Check if required columns exist
  required_cols <- c("degree_std", "betweenness_std", "poverty_pct", 
                     "poptot", "borough", "Longitude", "Latitude",
                     "felony_count", "misdemeanor_count", "violation_count", "total_crime_count")
  
  missing_cols <- required_cols[!required_cols %in% colnames(adata)]
  if (length(missing_cols) > 0) {
    log_message(paste("ERROR: Missing required columns:", paste(missing_cols, collapse = ", ")))
    
    # Attempt to create missing columns where possible
    if ("borough" %in% missing_cols && "Borough" %in% colnames(adata)) {
      adata$borough <- adata$Borough
      log_message("Created 'borough' from 'Borough'")
      missing_cols <- missing_cols[missing_cols != "borough"]
    }
    
    if ("degree_std" %in% missing_cols && "Degree_Centrality" %in% colnames(adata)) {
      adata$degree_std <- scale(adata$Degree_Centrality)[,1]
      log_message("Created 'degree_std' from 'Degree_Centrality'")
      missing_cols <- missing_cols[missing_cols != "degree_std"]
    }
    
    if ("betweenness_std" %in% missing_cols && "Betweenness_Centrality" %in% colnames(adata)) {
      adata$betweenness_std <- scale(adata$Betweenness_Centrality)[,1]
      log_message("Created 'betweenness_std' from 'Betweenness_Centrality'")
      missing_cols <- missing_cols[missing_cols != "betweenness_std"]
    }
    
    if (length(missing_cols) > 0) {
      stop(paste("Cannot proceed without columns:", paste(missing_cols, collapse = ", ")))
    }
  }
  
  # Data preparation and cleaning
  adata <- adata %>%
    filter(poptot > 0) %>%  # Remove areas with zero population
    drop_na(degree_std, betweenness_std, poverty_pct, poptot, borough, Longitude, Latitude,
            felony_count, misdemeanor_count, violation_count, total_crime_count)
  
  log_message(paste("Cleaned dataset contains", nrow(adata), "observations"))
  
  # Convert borough to factor
  adata$borough <- factor(adata$borough)
  
  # Create log population for offset
  if (!"log_poptot" %in% colnames(adata)) {
    adata$log_poptot <- log(adata$poptot)
    log_message("Created log_poptot for offset term")
  }
  
  # --------------------------------------------------------
  # 2. MULTICOLLINEARITY CHECK
  # --------------------------------------------------------
  log_message("Checking for multicollinearity")
  vif_results <- vif(lm(degree_std ~ betweenness_std + poverty_pct, data = adata))
  log_message("Variance Inflation Factors (VIF):")
  print(vif_results)
  
  # Write VIF results to file
  write.csv(data.frame(Variable = names(vif_results), VIF = vif_results), 
            file = file.path(output_dir, "vif_results.csv"), 
            row.names = FALSE)
  
  # --------------------------------------------------------
  # 3. MODEL DEFINITION
  # --------------------------------------------------------
  log_message("Defining model formula")
  
  parsimonious_formula <- function(crime_var) {
    as.formula(paste(
      crime_var,
      "~ degree_std + betweenness_std + ",
      "s(poverty_pct, k=15) + borough + ",
      "ti(poverty_pct, Longitude, Latitude, k=c(10,30)) + ",
      "offset(log_poptot)"
    ))
  }
  
  # --------------------------------------------------------
  # 4. MODEL FITTING
  # --------------------------------------------------------
  log_message("Fitting models for different crime types")
  
  crime_types <- c("felony_count", "misdemeanor_count", "violation_count", "total_crime_count")
  
  # Initialize lists and dataframes for results
  pars_models <- list()
  pars_results <- data.frame(
    crime_type = character(),
    AIC = numeric(),
    deviance_explained = numeric(),
    stringsAsFactors = FALSE
  )
  
  parametric_results <- data.frame(
    crime_type = character(),
    term = character(),
    estimate = numeric(),
    std_error = numeric(),
    t_value = numeric(),
    p_value = numeric(),
    effect_size_pct = numeric(),
    stringsAsFactors = FALSE
  )
  
  smooth_results <- data.frame(
    crime_type = character(),
    term = character(),
    edf = numeric(),
    ref_df = numeric(),
    f_value = numeric(),
    p_value = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Fit models for each crime type
  for (crime in crime_types) {
    log_message(paste("Fitting model for", crime))
    
    # Fit model
    mod <- gam(
      formula = parsimonious_formula(crime),
      data = adata,
      family = mgcv::nb(),
      method = "REML"
    )
    
    # Store model
    pars_models[[crime]] <- mod
    
    # Extract model summary
    mod_summary <- summary(mod)
    
    # Store model performance metrics
    pars_results <- rbind(
      pars_results,
      data.frame(
        crime_type = crime,
        AIC = AIC(mod),
        deviance_explained = mod_summary$dev.expl,
        stringsAsFactors = FALSE
      )
    )
    
    # Extract parametric coefficients
    param_coefs <- mod_summary$p.table
    param_data <- data.frame(
      crime_type = crime,
      term = rownames(param_coefs),
      estimate = param_coefs[, "Estimate"],
      std_error = param_coefs[, "Std. Error"],
      t_value = param_coefs[, "t value"],
      p_value = param_coefs[, "Pr(>|t|)"],
      stringsAsFactors = FALSE
    )
    
    # Calculate effect sizes as percentage change
    param_data$effect_size_pct <- ifelse(
      param_data$term %in% c("degree_std", "betweenness_std") | grepl("^borough", param_data$term),
      (exp(param_data$estimate) - 1) * 100,
      NA
    )
    
    parametric_results <- rbind(parametric_results, param_data)
    
    # Extract smooth terms
    smooth_coefs <- mod_summary$s.table
    smooth_data <- data.frame(
      crime_type = crime,
      term = rownames(smooth_coefs),
      edf = smooth_coefs[, "edf"],
      ref_df = smooth_coefs[, "Ref.df"],
      f_value = smooth_coefs[, "F"],
      p_value = smooth_coefs[, "p-value"],
      stringsAsFactors = FALSE
    )
    
    smooth_results <- rbind(smooth_results, smooth_data)
    
    # Run diagnostic checks
    pdf(file.path(output_dir, paste0(crime, "_diagnostics.pdf")))
    par(mfrow=c(2,2))
    gam.check(mod)
    dev.off()
    
    log_message(paste("Completed model for", crime))
  }
  
  # --------------------------------------------------------
  # 5. SAVE RESULTS
  # --------------------------------------------------------
  log_message("Saving model results")
  
  # Save model summary statistics
  write.csv(pars_results, file = file.path(output_dir, "model_performance.csv"), row.names = FALSE)
  
  # Save parametric coefficients
  write.csv(parametric_results, file = file.path(output_dir, "parametric_coefficients.csv"), row.names = FALSE)
  
  # Save smooth term results
  write.csv(smooth_results, file = file.path(output_dir, "smooth_terms.csv"), row.names = FALSE)
  
  # Create a summary table for the paper
  coef_summary <- parametric_results %>%
    filter(term %in% c("degree_std", "betweenness_std") | grepl("^borough", term)) %>%
    select(crime_type, term, estimate, p_value, effect_size_pct) %>%
    pivot_wider(
      names_from = crime_type,
      values_from = c(estimate, p_value, effect_size_pct)
    )
  
  write.csv(coef_summary, file = file.path(output_dir, "coefficient_summary.csv"), row.names = FALSE)
  
  # Save smooth term summary for the paper
  smooth_summary <- smooth_results %>%
    select(crime_type, term, edf, p_value) %>%
    pivot_wider(
      names_from = crime_type,
      values_from = c(edf, p_value)
    )
  
  write.csv(smooth_summary, file = file.path(output_dir, "smooth_summary.csv"), row.names = FALSE)
  
  # Save model objects for potential later use
  saveRDS(pars_models, file = file.path(output_dir, "gam_models.rds"))
  
  # --------------------------------------------------------
  # 6. CREATE VISUALIZATION FOR EFFECT SIZES
  # --------------------------------------------------------
  log_message("Creating effect size visualizations")
  
  # Filter for coefficients of interest and prepare for plotting
  effect_size_data <- parametric_results %>%
    filter(term %in% c("degree_std", "betweenness_std") | grepl("^borough", term)) %>%
    mutate(
      term_group = case_when(
        term %in% c("degree_std", "betweenness_std") ~ "Network Centrality",
        grepl("^borough", term) ~ "Borough Effects",
        TRUE ~ "Other"
      ),
      term_label = case_when(
        term == "degree_std" ~ "Degree Centrality",
        term == "betweenness_std" ~ "Betweenness Centrality",
        term == "boroughBronx" ~ "Bronx vs. Brooklyn",
        term == "boroughM" ~ "Manhattan vs. Brooklyn",
        term == "boroughQ" ~ "Queens vs. Brooklyn",
        TRUE ~ term
      ),
      significant = p_value < 0.05
    )
  
  # Save effect size data for plotting
  write.csv(effect_size_data, file = file.path(output_dir, "effect_size_data.csv"), row.names = FALSE)
  
  # Create effect size plots for each crime type
  for (crime in crime_types) {
    crime_effects <- effect_size_data %>% 
      filter(crime_type == crime) %>%
      arrange(term_group, effect_size_pct)
    
    # Skip if no data
    if (nrow(crime_effects) == 0) next
    
    # Create the plot
    p <- ggplot(crime_effects, aes(x = reorder(term_label, effect_size_pct), y = effect_size_pct, fill = significant)) +
      geom_col() +
      coord_flip() +
      labs(
        title = paste("Effect Sizes for", gsub("_count", "", crime)),
        x = "",
        y = "Percentage Change in Expected Crime Count",
        fill = "Statistically\nSignificant"
      ) +
      scale_fill_manual(values = c("TRUE" = "#1a9641", "FALSE" = "#bababa")) +
      theme_minimal() +
      theme(
        legend.position = "bottom",
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold")
      ) +
      facet_grid(term_group ~ ., scales = "free_y", space = "free")
    
    # Save the plot
    ggsave(
      file.path(output_dir, paste0(crime, "_effect_sizes.png")),
      plot = p,
      width = 8,
      height = 6,
      dpi = 300
    )
  }
  
  # Create a combined effect size plot for degree centrality across crime types
  degree_effects <- effect_size_data %>%
    filter(term == "degree_std") %>%
    mutate(crime_type = gsub("_count", "", crime_type))
  
  if (nrow(degree_effects) > 0) {
    p_degree <- ggplot(degree_effects, aes(x = reorder(crime_type, effect_size_pct), y = effect_size_pct, fill = crime_type)) +
      geom_col() +
      labs(
        title = "Effect of Degree Centrality on Crime",
        x = "Crime Type",
        y = "Percentage Increase in Expected Crime Count",
        fill = "Crime Type"
      ) +
      scale_fill_viridis_d() +
      theme_minimal() +
      theme(
        legend.position = "none",
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold")
      )
    
    ggsave(
      file.path(output_dir, "degree_centrality_effects.png"),
      plot = p_degree,
      width = 8,
      height = 5,
      dpi = 300
    )
  }
  
  log_message("Statistical analysis completed successfully")
  return(pars_models)
}

# Execute the analysis if run directly
if (sys.nframe() == 0) {
  run_statistical_analysis()
}
