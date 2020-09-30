
#---------- Setup ----------#

library(tidyverse)
library(readxl)
theme_set(theme_classic())
theme_set(theme_classic(base_size = 12))
setwd("~/Documents/flexible-routing")


#---------- Dataset Preparation ----------#

# Get simulation output files
files <- list.files('output', pattern = '.xlsx')

# Import and combine datasets with individual runs
sims <- data.frame()
for (i  in 1:length(files)){
   df <- read_xlsx(paste('output/', files[i], sep = ''), sheet = 1)
   sims <- rbind(sims, df)
}
colnames(sims) = c('ID','Scenario','Customers','Strategy','Metric','Value')

# Rename scenarios
sims[(sims$Scenario == 'baseline'),]$Scenario = 'Baseline'
sims[(sims$Scenario == 'baseline_k3'),]$Scenario = 'Medium Overlap'
sims[(sims$Scenario == 'baseline_k1'),]$Scenario = 'Small Overlap'
sims[(sims$Scenario == 'short_route'),]$Scenario = 'Short Route'
sims[(sims$Scenario == 'long_route'),]$Scenario = 'Long Route'
sims[(sims$Scenario == 'stochastic_customers'),]$Scenario = 'Stoch. Cust.'

# Drop full flexibility strategy
sims <- sims[!(sims$Strategy == 'fully flexible'),]

# Rename routing strategies
sims[(sims$Strategy == 'dedicated'),]$Strategy = 'Dedicated'
sims[(sims$Strategy == 'overlapped'),]$Strategy = 'Overlapped'
sims[(sims$Strategy == 'reoptimization'),]$Strategy = 'Reoptimization'

# Rename metrics
sims[(sims$Metric == 'total cost'),]$Metric = 'Total Cost'
sims[(sims$Metric == 'circular cost'),]$Metric = 'Circular Cost'
sims[(sims$Metric == 'radial cost'),]$Metric = 'Radial Cost'
sims[(sims$Metric == 'trip count'),]$Metric = 'Trip Count'


#---------- Baseline Graphs ----------#

# Total cost
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Total Cost') %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Strategy, color = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 1) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   scale_color_grey(start = 0.3)

ggsave('figures/total_cost.png')


# Relative to Reoptimization
reopt <- sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy == 'Reoptimization') %>%
   group_by(Customers) %>%
   summarise(avg_reopt = mean(Value))

sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost') %>%
   merge(reopt) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)/mean(avg_reopt)) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Strategy, color = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 1) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost (Rel. to Reoptimization)') + 
   expand_limits(y=0) +
   scale_color_grey(start = 0.3)
   
ggsave('figures/rel_cost.png')


# Circular and Radial Cost
sims %>%
   filter(Scenario == 'Baseline',
          Metric %in% c('Circular Cost', 'Radial Cost')) %>%
   group_by(Customers, Strategy, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Metric, fill = Metric) +
   geom_area() +
   labs(x = 'Number of Customers', y = 'Cost') + 
   facet_wrap(Strategy ~.) +
   scale_fill_grey(start = 0.4)

ggsave('figures/cost_breakdown.png')


# Number of trips
# KL: Include condidence bars?
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Trip Count') %>%
   mutate(Customers = factor(Customers)) %>%
   ggplot() +
   aes(x = Customers, y = Value, fill = Strategy) + 
   geom_bar(stat = 'identity', position = 'dodge') +
   labs(x = 'Number of Customers', y = 'Trip Count') + 
   scale_fill_grey(start = 0.3)

ggsave('figures/trips.png')

# Distribution of individual runs' costs by strategy
sims %>%
   filter(Scenario == 'Baseline',
          Customers %in% c(5,20,80),
          Metric == 'Total Cost') %>%
   mutate(Customers = factor(Customers)) %>%
   ggplot() +
   aes(x = Value, fill=Customers) +
   geom_histogram(color='black', bins = 100) + 
   facet_grid(Strategy~.) +
   labs(x = 'Total Cost', y = 'Count') + 
   scale_fill_grey(start = 0.3, end = 0.9)

ggsave('figures/hist_total.png')



#---------- Scenario Comparisons ----------#

# Compare k=1, k=3, and k=5 strategies
# TODO


# Comparison of scenarios' distributions within individual strategies
strategies = unique(sims$Strategy)
for (strat in strategies){
   
   sims %>%
      filter(!Scenario %in% c('Medium Overlap', 'Long Route'),
             Customers %in% c(10,80),
             Strategy == strat,
             Metric == 'Total Cost') %>%
      mutate(Customers = factor(Customers)) %>%
      ggplot() +
      aes(x = Value, fill=Customers) +
      geom_histogram(color='black', bins = 75) + 
      facet_grid(Scenario~.) +
      labs(x = 'Total Cost', y = 'Count') + 
      scale_fill_grey(start = 0.4)
   
   ggsave(paste('figures/hist_compare_',strat,'.png', sep = ''))
   
}



