# 加载所需的库
library(randomForest)
 
library(network)
library(sna)
library(intergraph)

# 设置颜色映射函数
color_map <- function(mat) {
  # 设定一个颜色向量，颜色将反映相关性的强度
  max_val <- max(abs(mat), na.rm = TRUE)
  color_palette <- colorRampPalette(c("blue", "white", "red"))(100)
  color_index <- as.integer((abs(mat) / max_val) * 99) + 1
  color_palette[color_index]
}

# 处理 PM25 数据集
process_data <- function(data, response_var) {
  rf_model <- randomForest(as.formula(paste(response_var, "~ .")), data = data, mtry = 3, importance = TRUE, proximity = FALSE)
  set.seed(1701)
  vivi_rf <- vivi(fit = rf_model, 
                  data = data, 
                  response = response_var,
                  gridSize = 50,
                  importanceType = "agnostic",
                  nmax = 500,
                  reorder = TRUE,
                  predictFun = NULL,
                  numPerm = 4,
                  showVimpError = FALSE)
  
  # 生成热图
  heatmap_colors <- color_map(vivi_rf)
  viviHeatmap(mat = vivi_rf, col = heatmap_colors)
  
  # 生成网络图
  layout <- cbind(c(1,1,1,1,2,2,2,2,2), c(1,2,4,5,1,2,3,4,5))
  viviNetwork(mat = vivi_rf, intThreshold = 0.12, removeNode = FALSE, layout = layout)
  viviNetwork(mat = vivi_rf, intThreshold = 0.12, removeNode = TRUE, layout = layout)
}

# 处理 PM25 数据
process_data(PM25OSLO, "PM25")
process_data(PM25Bergen, "PM25")
process_data(PM25Trondheim, "PM25")

# 处理 NOx 数据
process_data(NOxOSLO, "NOx")
process_data(NOxBergen, "NOx")
process_data(NOxTrondheim, "NOx")

###Done
