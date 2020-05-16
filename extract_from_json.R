# load package to manipulate data
library(tidyverse)

# load package to process json files
library(jsonlite)

# where the json file is
dest_folder <- "/Users/oliviergimenez/Desktop/36.2_G_Lot3resized/"

# read in the json file
pixjson <- fromJSON(paste0(dest_folder,'box_ain.json'), flatten = TRUE)

# what structure?
str(pixjson)

# names
names(pixjson)

# categories are animal, human and vehicules
pixjson$detection_categories

# get pix only
pix <- pixjson$images
names(pix)

# unlist detections and bbox and store everything in a csv file
pix %>% 
  as_tibble() %>%
  unnest(detections) %>%
  unnest_wider(bbox) %>%
  rename(xmin = '...1',
         xmax = '...2',
         width = '...3',
         height = '...4') %>%
  write_csv('testain.csv')
