library(ggplot2)
library(ggforce)
library(latex2exp)

if (!("Figs" %in% list.files())) dir.create("Figs")

# Create toy data
Y  <- c(1, 1, 1,   1,   1,   0,   2,   0,   2,   0,   2,  1)
X1 <- c(1, 2, 5,   6,   3.5, 4,   2,   5.5, 2.5, 6.5, 1,  4)
X2 <- c(1, 2, 2.5, 3.5, 3,   5.5, 4.5, 5.5, 6,   7,   5,  1)
data <- data.frame(cbind(X1, X2, Y))

# Set ggplot "theme" for plots
mytheme <- theme(axis.title = element_text(size = 15),
                 legend.title = element_text(size = 14),
                 legend.title.align = 0.5,
                 legend.text = element_text(size = 13),
                 legend.background = element_rect(color = "black"),
                 legend.position = c(0.925, 0.15))

# Plot toy data w/ question mark for unlabeled observation
pdf("Figs/toy_data.pdf", width = 4.66, height = 5.83)
ggplot() +
  geom_point(data = data, aes(x = X1, y = X2, col = factor(Y)), size = 2.5) +
  geom_point(aes(x = 1, y = 6), shape = "?", size = 4.5) +
  labs(x = TeX("X_1"), y = TeX("X_2"), col = "Y") +
  scale_x_continuous(limits = c(0, 7), breaks = 0:7) +
  scale_y_continuous(limits = c(0, 7), breaks = 0:7) +
  theme_classic() + mytheme
dev.off()

# Include new observation (circled, blue) and plot splits as lines
data <- rbind(data, c(1, 6, 2))

pdf("Figs/toy_data_splits.pdf", width = 4.66, height = 5.83)
ggplot() +
  geom_point(data = data, aes(x = X1, y = X2, col = factor(Y)), size = 2.5) +
  geom_circle(aes(x0 = 1, y0 = 6, r = 0.15)) +
  geom_hline(yintercept = 4, linetype = "dashed") +
  geom_vline(xintercept = 3, linetype = "dotted") + 
  labs(x = TeX("X_1"), y = TeX("X_2"), col = "Y") +
  scale_x_continuous(limits = c(0, 7), breaks = 0:7) +
  scale_y_continuous(limits = c(0, 7), breaks = 0:7) +
  theme_classic() + mytheme
dev.off()

# Plot splits from final decision tree on the original data
data <- data[-nrow(data), ]

pdf("Figs/toy_data_splits_full.pdf", width = 4.66, height = 5.83)
ggplot() +
  geom_point(data = data, aes(x = X1, y = X2, col = factor(Y)), size = 2.5) +
  geom_hline(yintercept = 3.5, linetype = "dashed") +
  geom_vline(xintercept = 2.5, linetype = "dotted") + 
  scale_x_continuous(limits = c(0, 7), breaks = 0:7) +
  scale_y_continuous(limits = c(0, 7), breaks = 0:7) +
  labs(x = TeX("X_1"), y = TeX("X_2"), col = "Y") +
  theme_classic() + mytheme
dev.off()
       