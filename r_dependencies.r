# Install required R packages if not already installed

# List of required packages
required_packages <- c(
  "tidyverse",  # For data manipulation and visualization
  "mgcv",       # For GAM models
  "MASS",       # For negative binomial models
  "viridis",    # For color palettes
  "sf",         # For spatial data handling
  "corrplot",   # For correlation visualization
  "gridExtra",  # For arranging multiple plots
  "caret",      # For model validation
  "ROCR",       # For ROC curves
  "car"         # For diagnostic functions
)

# Check which packages are not installed
new_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]

# Install missing packages
if(length(new_packages)) {
  install.packages(new_packages, repos = "https://cran.rstudio.com/")
}

# Print confirmation message
cat("Required R packages check completed.\n")
for(pkg in required_packages) {
  if(pkg %in% installed.packages()[,"Package"]) {
    cat(sprintf("✓ %s is installed.\n", pkg))
  } else {
    cat(sprintf("✗ Failed to install %s.\n", pkg))
  }
}
