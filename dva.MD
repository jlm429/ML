---
title: 'Data and Visual Analytics'
subtitle: |-
  
output:
  RMD (R Markdown) 
---

> Professional Education by State (Midwest Data Set)

```{r}

library(dplyr)
library(plyr)
library(magrittr)
library(ggplot2)
library(ggmap)
library(e1071)

#1

#ordered boxplots
ggplot(midwest, aes(reorder(state, percprof, median), percprof)) + geom_boxplot() + coord_flip() + scale_x_discrete("State") + ylab("Ordered Median Percent Professional")
ggplot(midwest, aes(reorder(state, percprof, mean), percprof)) + geom_boxplot() + coord_flip() + scale_x_discrete("State") + ylab("Ordered Mean Percent Professional")

#histograms overlay
ggplot(midwest,aes(x=percprof)) + 
     geom_histogram(data=subset(midwest,state == 'IL'),fill = "red", alpha = 0.2) +
     geom_histogram(data=subset(midwest,state == 'MI'),fill = "blue", alpha = 0.2) +
     geom_histogram(data=subset(midwest,state == 'IN'),fill = "green", alpha = 0.2) +
     geom_histogram(data=subset(midwest,state == 'WI'),fill = "yellow", alpha = 0.2) +
     geom_histogram(data=subset(midwest,state == 'OH'),fill = "purple", alpha = 0.2) +
     xlab("Alpha Blending - Percprof by State")

#facet
ggplot(midwest, aes(percprof)) +
    geom_histogram() +
    facet_wrap(~state, nrow = 1)	 

#heat map from ggplot p.54
il_counties <- map_data("county", "illinois") %>% select(lon = long, lat, group, id = subregion)

il_census <- midwest %>%
    tbl_df() %>%
    filter(state == "IL") %>%
    mutate(county = tolower(county)) %>%
    select(county, area, poptotal, popadults, percwhite, percblack, percollege, perchsd, percprof, percbelowpoverty)		 

census_counties <- left_join(il_census, il_counties, by = c("county" = "id"))

ggplot(census_counties, aes(lon, lat, group = county)) +
    geom_polygon(aes(fill = percprof)) +
    coord_quickmap()
	
#weighted mean (repeat for each state)
weighted.mean(il_census$percprof, il_census$poptotal)

#2

#ggpairs
DF = midwest[, c("perchsd", "percollege", "state")]
ggpairs(DF)


#heatmap of 2d bin counts		
#from ggplot p. 39
mod_coef <- coef(lm(log10(perchsd) ~ log10(percollege), data = midwest))
ggplot(midwest, aes(log10(percollege), log10(perchsd))) +
     geom_bin2d() +
     geom_abline(intercept = mod_coef[1], slope = mod_coef[2],
                 colour = "white", size = 1) +
     facet_wrap(~state, nrow = 1)
		
```

![Heatmap](https://github.com/jlm429/ML/blob/master/images/dvaILpercprofmap.png)  

![Box Plot](https://github.com/jlm429/ML/blob/master/images/dvaBoxPlot.png)   

![Alpha Blending](https://github.com/jlm429/ML/blob/master/images/dvaPercprofHist.png)   
