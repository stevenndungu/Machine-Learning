library(ggplot2)
my_scatterplot <- function(dat, x_c, y_c) {
  scatterpl <- ggplot(dat, aes(x_c, y_c)) + geom_point() +
    geom_abline() +
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
  return(scatterpl)
}
my_scatterplot(diamonds, diamonds$x, diamonds$y)

#Boxplot with a numeric and factor variable


my_boxplot <- function(dat, fill_c, y_c) {
  boxplott <- ggplot(data = dat,mapping = aes(y = y_c, fill = fill_c,x = "")) +
    geom_boxplot() +
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
    theme_grey()
  return(boxplott)
}
my_boxplot(diamonds, diamonds$cut, diamonds$y)

#Boxplot with a numeric and 2 factor variables

my_boxplot <- function(dat, x_c, y_c, fill_c) {
  boxplott <- ggplot(data = dat,mapping = aes(y = y_c, fill = fill_c, x = x_c)) +
    geom_boxplot() +
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
    theme_grey()
  return(boxplott)
}
my_boxplot(diamonds, diamonds$cut, diamonds$y, diamonds$color)


#Barplot of one variable

my_barplot <- function(dat, x_c) {
  barplott <- ggplot(data = dat, mapping = aes(x = x_c)) +
    theme(
      axis.text.x = element_text(
        family = "Times",
        angle = 45,
        colour = "black",
        face = "bold"
      )
    ) +
    theme(
      axis.text.x = element_text(
        colour = "black",
        face = "bold",
        family = "Times",
        angle = 45
      )
    ) +
    geom_bar() + 
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
  return(barplott)
}

my_barplot(diamonds, diamonds$cut)

#Barplot with more than two variables
my_barplot <- function(dat, x_c, fill_c) {
  barplott <- ggplot(data = dat,mapping = aes(x = x_c, fill = fill_c)) +
    theme(
      axis.text.x = element_text(
        family = "Times",
        angle = 45,
        colour = "blue",
        face = "bold"
      )
    ) +
    theme(
      axis.text.x = element_text(
        colour = "blue",
        face = "bold",
        family = "Times",
        angle = 45
      )
    ) +
    geom_bar() + 
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
  return(barplott)
}
my_barplot(diamonds, diamonds$cut, diamonds$color)


#Barplot with one factor and a numeric variable
my_barplot <- function(dat, x_c, y_c) {
  barplott <-
    ggplot(data = dat, mapping = aes(x = x_c, y = y_c)) +
    theme(
      axis.text.x = element_text(
        family = "Times",
        angle = 45,
        colour = "blue",
        face = "bold"
      )
    ) +
    theme(
      axis.text.x = element_text(
        colour = "blue",
        face = "bold",
        family = "Times",
        angle = 45
      )
    ) +
    geom_bar(stat = "identity", position = position_dodge()) +
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
  return(barplott)
}
my_barplot(diamonds, diamonds$cut, diamonds$carat)


#Barplot with two factor and a numeric variable
my_barplot <- function(dat, x_c, y_c, fill_c) {
  barplott <-
    ggplot(data = dat, mapping = aes(x = x_c, y = y_c, fill = fill_c)) +
    theme(
      axis.text.x = element_text(
        family = "Times",
        angle = 45,
        colour = "blue",
        face = "bold"
      )
    ) +
    theme(
      axis.text.x = element_text(
        colour = "blue",
        face = "bold",
        family = "Times",
        angle = 45
      )
    ) +
    geom_bar(stat = "identity", position = position_dodge()) +
    ggtitle(label= "") +
    xlab(label= "") +
    ylab(label = "")
  #geom_text(aes(x_c, y_c, label = y_c),
  #vjust = -0.3,
  # colour = "red")
  return(barplott)
}
my_barplot(diamonds, diamonds$cut, diamonds$carat, diamonds$color)


#Construct a barplot whith sorted bars

table <- table(diamonds$color)
table <- as.data.frame(table)
names(table) <- c("Colour", "Count")
table <- transform(table,
                   Colour = reorder(Colour, Count))

plot <-
  ggplot(data = table) + geom_bar(aes(Colour, Count), stat = "identity") +
  geom_text(aes(Colour, Count, label = Count),
            hjust = -0.3,
            colour = "red") +
  coord_flip() + 
  ggtitle(label= "") +
  xlab(label= "") +
  ylab(label = "")
  theme_grey()
plot

#Histogram count

barfill <- "#4271AE"
barlines <- "#1F3552"

my_histgram <- function(dat, x_c) {
hist <- ggplot(dat, aes(x = x_c)) +
  geom_histogram(aes(y = ..count..), bins = 30,
                 colour = barlines, fill = barfill) +
  scale_x_continuous(name = "Mean ozone")+
                     #breaks = seq(0, 175, 25),
                     #limits=c(0, 175)) +
  scale_y_continuous(name = "Count") +
  ggtitle("")
return(hist)
}
my_histgram(diamonds, diamonds$carat)


#Histogram Density

barfill <- "#4271AE"
barlines <- "#1F3552"

my_histgram <- function(dat, x_c) {
  hist <- ggplot(dat, aes(x = x_c)) +
    geom_histogram(aes(y = ..density..), bins = 30,
                   colour = barlines, fill = barfill) +
    scale_x_continuous(name = "Mean ozone")+
    #breaks = seq(0, 175, 25),
    #limits=c(0, 175)) +
    scale_y_continuous(name = "Density") +
    geom_density(adjust=3)
    ggtitle("")
  return(hist)
}
my_histgram(diamonds, diamonds$carat)



#Histogram Density

barfill <- "#4271AE"
barlines <- "#1F3552"

my_histgram <- function(dat, x_c) {
  hist <- ggplot(dat, aes(x = x_c)) +
    geom_histogram(aes(y = ..density..), bins = 30,
                   colour = barlines, fill = barfill) +
    scale_x_continuous(name = "Mean ozone")+
    #breaks = seq(0, 175, 25),
    #limits=c(0, 175)) +
    scale_y_continuous(name = "Density") +
    geom_density(adjust=3)
  ggtitle("")
  return(hist)
}
my_histgram(diamonds, diamonds$carat)



# Correlations

 #devtools::install_github("kassambara/ggcorrplot")
library(ggplot2)
library(ggcorrplot)

# Correlation matrix
data(mtcars)
corr <- round(cor(mtcars), 1)

# Plot
ggcorrplot(corr, hc.order = TRUE, 
           type = "lower", 
           lab = TRUE, 
           lab_size = 3, 
           method="circle", 
           colors = c("tomato2", "white", "springgreen3"), 
           title="Correlogram of mtcars", 
           ggtheme=theme_bw)
