# Loading libraries
library(fclust)
library(ppclust)
library(SoftClustering)
library(mclust)
library(funtimes)
library(Rfast)
library("RColorBrewer")

#################################################################################
# Rough K-Means
#################################################################################

# Rough K-Means implementation
rough_kmeans = function(dataset, means_matrix = 2, num_clusters = 2,
                        max_iterations = 100, threshold = 1.5, 
                        weight_lower = 0.7) {
  
  # Setting of variables and matrices
  num_obs = nrow(dataset)
  num_features = ncol(dataset)
  threshold = threshold^2
  dataset = as.matrix(dataset)
  
  # Initialize Upper Approximation (UA) matrix and means
  old_upper_approx_matrix = matrix(999, nrow = num_obs, ncol = num_clusters)
  means_matrix = initialize_means(dataset, num_clusters, means_matrix)
  upper_approx_matrix = assign_upper_approx(dataset, means_matrix, threshold)
  
  # Initialize the iteration
  iterations = 0
  
  # Repeat until classification unchanged or max_iterations reached
  while ( !identical(old_upper_approx_matrix, upper_approx_matrix) 
          && iterations < max_iterations ) {
    
    # Lower Approximation (LA) Matrix and UA-LA matrix (boundary)
    lower_approx_matrix = assign_lower_approx(upper_approx_matrix)
    boundary_matrix = upper_approx_matrix - lower_approx_matrix
    
    # Calculate sums of observations in LA and boundary in every cluster
    means_matrix_lower = crossprod(lower_approx_matrix, dataset)
    means_matrix_boundary = crossprod(boundary_matrix, dataset) 
    
    # Update means matrix
    for (i in 1:num_clusters) {
      
      # Dividers means calculation, cardinalities of LA and UA-LA (boundary)
      divider_lower_approx = sum(lower_approx_matrix[, i])
      divider_boundary = sum(boundary_matrix[, i])
      
      if (divider_lower_approx != 0 && divider_boundary != 0) {
        means_matrix_lower[i,] = means_matrix_lower[i,] / 
                                 divider_lower_approx
        means_matrix_boundary[i,] = means_matrix_boundary[i,] / 
                                    divider_boundary
        means_matrix[i,] = weight_lower*means_matrix_lower[i,] + 
                           (1-weight_lower)*means_matrix_boundary[i,]
      } else if (divider_boundary == 0) {
        means_matrix[i,] = means_matrix_lower[i,] / divider_lower_approx
      } else { # if(divider_lower_approx[,i]) == 0)
        means_matrix[i,] = means_matrix_boundary[i,] / divider_boundary
      }
    }
    
    # Saving upper approximations of previous iteration
    old_upper_approx_matrix = upper_approx_matrix
    upper_approx_matrix = assign_upper_approx(dataset, means_matrix, 
                                              threshold)
    
    iterations = iterations + 1
  }

  return ( list(upper_approx=upper_approx_matrix, cluster_means=means_matrix, 
                num_iterations=iterations) )
}

#################################################################################
# initialize_means
#################################################################################

# Delivers an initial means matrix
initialize_means = function(data_matrix, num_clusters, means_matrix) {
  
  if(is.matrix(means_matrix)) { # means pre-defined # no action required
    
  }else if (means_matrix == 1) {  # random means
    
    num_features = ncol(data_matrix)
    means_matrix = matrix(0, nrow=num_clusters, ncol=num_features)
    for (i in 1:num_features) {
      means_matrix[,i] = c(runif(num_clusters, min(data_matrix[,i]), 
                                 max(data_matrix[,i])))
    }
    
  }else if (means_matrix == 2) { # maximum distance means
    
    means_objects = seq(length=num_clusters, from=0, by=0)
    objects_dist_matrix = as.matrix(dist(data_matrix))
    
    pos_vector = which(objects_dist_matrix == max(objects_dist_matrix),
                       arr.ind = TRUE)
    means_objects[1] = pos_vector[1,1]
    means_objects[2] = pos_vector[1,2]
    
    for(i in seq(length=(num_clusters-2), from=3, by=1) ) {
      means_objects[i] = which.max( 
        colSums(objects_dist_matrix[means_objects, -means_objects]))
    }
    
    means_matrix = data_matrix[means_objects, ]
    
  }
  
  return(as.matrix(means_matrix))
}

#################################################################################
# assign_upper_approx
#################################################################################

# Assign object to upper approximation
assign_upper_approx = function(dataset, means_matrix, threshold) {
  
  num_obs  = nrow(dataset)
  num_clusters = nrow(means_matrix)
  
  distances_to_clusters = seq(length=num_clusters, from=0, by=0)
  
  upper_approx_matrix = matrix(0, nrow = num_obs, ncol = num_clusters )
  
  for (i in 1:num_obs) {
    
    # distances_to_clusters from object i to all clusters j
    for (j in 1:num_clusters){
      distances_to_clusters[j] = sum( (dataset[i,] - means_matrix[j,] )^2 )
    }
    
    min_distance = max(min(distances_to_clusters), 1e-99)
    
    # Includes the closest objects.
    closest_clusters_idx = which((distances_to_clusters / min_distance)
                                 <= threshold)
    
    upper_approx_matrix[i, closest_clusters_idx] = 1
    
  }
  
  return(upper_approx_matrix)
}

#################################################################################
# assign_lower_approx
#################################################################################

# Assign object to lower approximation out of an upper approximation.
assign_lower_approx = function(upper_approx_matrix) {
  
  # Initialization of lower_approx_matrix
  lower_approx_matrix = 0 * upper_approx_matrix
  
  object_idx = which( rowSums(upper_approx_matrix) == 1 )
  
  lower_approx_matrix[object_idx, ] = upper_approx_matrix[object_idx, ]
  
  return(lower_approx_matrix)
}

#################################################################################
# Testing
#################################################################################

# Load datasets
data(DemoDataC2D2a)
data(iris)
g2 = read.delim("datasets/g2.txt", header = FALSE, sep = ",")
synth = read.delim("datasets/synth.txt", header = FALSE, sep = ",")

datasets = list(DemoDataC2D2a, g2, synth, iris[-5])
num_clusters = c(2, 2, 2, 3)

# Results
execution_times = matrix(nrow = 4, ncol = length(datasets), 
                  dimnames = list(c("Fuzzy K-Means", "Possibilistic K-Means", "Model-Based Clustering", "Rough K-Means"), 
                                  c("DemoDataC2D2a", "G2", "Synth", "Iris")))
silhouette_index = matrix(nrow = 4, ncol = length(datasets), 
                          dimnames = list(c("Fuzzy K-Means", "Possibilistic K-Means", "Model-Based Clustering", "Rough K-Means"), 
                                          c("DemoDataC2D2a", "G2", "Synth", "Iris")))

external_index = matrix(nrow = 4, ncol = 2, 
                          dimnames = list(c("Fuzzy K-Means", "Possibilistic K-Means", "Model-Based Clustering", "Rough K-Means"), 
                                          c("Purity", "ARI")))

#################################################################################
# Fuzzy K-Means
#################################################################################

for (i in 1:length(datasets)) {
  start_time = Sys.time()
  results_fkm = FKM(X=as.data.frame(datasets[i]), k=num_clusters[i], m=2)
  end_time = Sys.time()

  # Means
  results_fkm$H
  
  # Assignments
  results_fkm$U
  
  # Plots
  plot.fclust(x=results_fkm, umin=0.7, colclus=brewer.pal(n = 3, name = "Set1"))
  VIFCR(fclust.obj=results_fkm, which=1)
  
  # Validation
  execution_times[1, i] = end_time - start_time
  silhouette_index[1, i] = Fclust.index(results_fkm, "SIL.F")
}

#################################################################################
# Possibilistic K-Means
#################################################################################

for (i in 1:length(datasets)) {
  start_time = Sys.time()
  results_pcm = pcm(x=as.data.frame(datasets[i]), centers=num_clusters[i])
  end_time = Sys.time()
  
  # Means
  results_pcm$v
  
  # Assignments
  results_pcm$t
  
  sum(results_pcm$t < 0.5)
  
  # Conversion
  results_converted_pcm <- ppclust2(results_pcm, "fclust")
  results_converted_pcm$H = results_pcm$v
  
  results_converted_pcm$H
  results_converted_pcm$U
  
  # Plots
  plot.fclust(results_converted_pcm, umin=0.5, colclus=brewer.pal(n = 3, name = "Set1"))
  VIFCR(results_converted_pcm, 1)
  plotcluster(results_pcm, cp=1, cm="threshold", tv=0.5, trans=TRUE)
  
  # Validation
  execution_times[2, i] = end_time - start_time
  silhouette_index[2, i] = Fclust.index(results_converted_pcm, "SIL.F")
}

#################################################################################
# Model-Based Clustering - Based on Parameterized Finite Gaussian Mixture Models
#################################################################################

for (i in 1:length(datasets)) {
  start_time = Sys.time()
  results_mclust = Mclust(as.data.frame(datasets[i]), G=num_clusters[i])
  end_time = Sys.time()
  
  # Means
  results_mclust$parameters$mean
  
  # Assignments
  results_mclust$z
  
  results_fkm$H = t(results_mclust$parameters$mean)
  
  # Assignments
  results_fkm$U = results_mclust$z
  
  plot.fclust(results_fkm, umin=0.7, colclus=brewer.pal(n = 3, name = "Set1"))
  
  # Plot
  plot(results_mclust, what = c("classification", "uncertainty"), addEllipses = TRUE)
  
  # Validation
  execution_times[3, i] = end_time - start_time
  silhouette_index[3, i] = SIL.F(as.data.frame(datasets[i]), results_mclust$z)
}

#################################################################################
# Rough K-Means
#################################################################################

for (i in 1:length(datasets)) {
  start_time = Sys.time()
  results_rkm_custom = rough_kmeans(dataset=as.data.frame(datasets[i]), means_matrix=2, num_clusters=num_clusters[i], max_iterations=100, threshold=1.5, weight_lower=0.7)
  end_time = Sys.time()
  
  # Means
  results_rkm_custom$cluster_means
  
  # Assignments
  results_rkm_custom$upper_approx
  
  sum(rowSums(results_rkm_custom$upper_approx) > 1)
  
  # Plot
  SoftClustering:::plotRoughKMeans(as.data.frame(datasets[i]), upperMShipMatrix=results_rkm_custom$upper_approx, meansMatrix=results_rkm_custom$cluster_means, colouredPlot=TRUE)
  
  # Validation
  execution_times[4, i] = end_time - start_time
  silhouette_index[4, i] = SIL.F(as.data.frame(datasets[i]), results_rkm_custom$upper_approx)
}

#################################################################################
# Results
#################################################################################

# Metrics Visualization
print("Execution times (s):")
execution_times

print("Silhouette index:")
silhouette_index

# Calculation of External Indices with Iris dataset
external_index[1, 1] = purity(iris$Species, rowMaxs(results_fkm$U))$pur
external_index[1, 2] = adjustedRandIndex(iris$Species, rowMaxs(results_fkm$U))

external_index[2, 1] = purity(iris$Species, rowMaxs(results_pcm$t))$pur
external_index[2, 2] = adjustedRandIndex(iris$Species, rowMaxs(results_pcm$t))

external_index[3, 1] = purity(iris$Species, rowMaxs(results_mclust$z))$pur
external_index[3, 2] = adjustedRandIndex(iris$Species, rowMaxs(results_mclust$z))

external_index[4, 1] = purity(iris$Species, rowMaxs(results_rkm_custom$upper_approx))$pur
external_index[4, 2] = adjustedRandIndex(iris$Species, rowMaxs(results_rkm_custom$upper_approx))

print("External Indices Iris:")
external_index
