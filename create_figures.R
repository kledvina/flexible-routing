
#---------- Setup ----------#

library(tidyverse)
library(readxl)
library(RColorBrewer)
theme_set(theme_bw())
theme_set(theme_bw(base_size = 22))
setwd("~/Documents/flexible-routing")

num_sims <- 6000

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
sims[(sims$Scenario == 'low_capacity'),]$Scenario = 'Low Capacity'
sims[(sims$Scenario == 'high_capacity'),]$Scenario = 'High Capacity'

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

# Update ID to represent individual simulations
n_strat <- length(unique(sims$Strategy))
n_metric <- length(unique(sims$Metric))
sims <- sims %>%
   group_by(Scenario, Customers) %>%
   mutate(ID = rep(c(1:num_sims), each = n_strat*n_metric)) %>%
   ungroup()


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
   expand_limits(y=0) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   scale_color_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
   #scale_color_grey(start = 0.3)

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
   scale_color_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
   #scale_color_grey(start = 0.3)
   
ggsave('figures/rel_cost.png')

# Combined: Total & Relative to Reoptimization
sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost') %>%
   merge(reopt) %>%
   group_by(Customers, Strategy) %>%
   summarise(`Total Cost` = mean(Value),
             `Relative Cost` = mean(Value)/mean(avg_reopt)) %>%
   gather(Metric, Value, `Total Cost`, `Relative Cost`) %>%
   mutate(Metric = factor(Metric, levels = c("Total Cost", "Relative Cost"))) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Strategy, color = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 1) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   expand_limits(y=0) +
   facet_wrap(Metric ~., scales = 'free') +
   scale_color_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
#scale_color_grey(start = 0.3)
ggsave('figures/combined_total_rel_cost.png')


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
   scale_fill_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
   #scale_fill_grey(start = 0.4)

ggsave('figures/cost_breakdown.png')


# Number of trips
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Trip Count') %>%
   mutate(Customers = factor(Customers)) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y = Value, fill = Strategy) + 
   geom_bar(stat = 'identity', position = 'dodge') +
   labs(x = 'Number of Customers', y = 'Trip Count') + 
   scale_fill_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
   #scale_fill_grey(start = 0.3)

ggsave('figures/trips.png')


# Distribution of individual runs' costs by strategy
sims %>%
   filter(Scenario == 'Baseline',
          Customers %in% c(5,20,80),
          Metric == 'Total Cost') %>%
   mutate(`Number of Customers` = factor(Customers)) %>%
   ggplot() +
   aes(x = Value, fill=`Number of Customers`) +
   geom_histogram(color='black', bins = 50) + 
   facet_grid(Strategy~.) +
   labs(x = 'Total Cost', y = 'Count') + 
   scale_fill_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.25, legend.position="top")
   #scale_fill_grey(start = 0.3, end = 0.9)

ggsave('figures/hist_total.png')


# Percent of sims where overlapped did better than dedicated
sims %>%
   filter(Scenario == 'Baseline', Metric == 'Total Cost') %>%
   spread(Strategy, Value) %>%
   group_by(Scenario, Customers) %>%
   summarise(`Lower Cost` = 100*sum(Overlapped < Dedicated) / num_sims,
             `Equal Cost` = 100*sum(Overlapped == Dedicated) / num_sims,
             `Higher Cost` = 100*sum(Overlapped > Dedicated) / num_sims)



#---------- Scenario Comparisons ----------#
# Compare overlapped strategies for baseline, small overlap, and medium overlap
# scenarios to dedicated strategy (same for all scenarios)
sims %>%
   filter(Scenario %in% c('Baseline', 'Medium Overlap', 'Small Overlap'),
          Metric != 'Trip Count') %>%
   group_by(Scenario, Customers, Strategy, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ungroup() %>%
   spread(Strategy, Value) %>%
   mutate(Value = Overlapped / Dedicated,
          Customers = factor(Customers)) %>%
   ggplot() +
   aes(x = Customers, y = Value, fill = Scenario) + 
   geom_bar(stat = 'identity', position = 'dodge') +
   geom_hline(yintercept = 1.0, alpha = 0.75, linetype = 'dashed') +
   labs(x = 'Number of Customers', y = 'Cost (Rel. to Dedicated)') + 
   facet_wrap(Metric ~.) +
   theme(aspect.ratio = 0.75) +
   scale_fill_brewer(palette = "Dark2")
   #scale_fill_grey(start = 0.3)

ggsave('figures/overlap_size_cost.png')


# Compare low and high capacity scenarios to baseline
sims %>%
   filter(Scenario %in% c('Baseline', 'Low Capacity', 'High Capacity'),
          Metric == 'Total Cost') %>%
   mutate(Customers = factor(Customers)) %>%
   group_by(Scenario, Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   spread(Scenario, Value) %>%
   mutate(`High Capacity` = (`High Capacity`) / Baseline,
          `Low Capacity` = (`Low Capacity`) / Baseline) %>%
   gather(Scenario, Value, Baseline, `High Capacity`, `Low Capacity`) %>%
   filter(Scenario != 'Baseline') %>%
   ggplot() +
   aes(x = Customers, y =  Value, fill = Scenario) +
   geom_bar(stat = 'identity', position = 'dodge') +
   labs(x = 'Number of Customers', y = 'Cost (Rel. to Baseline)') + 
   facet_wrap(Strategy ~.) +
   theme(aspect.ratio = 0.75) +
   scale_fill_brewer(palette = "Dark2")
   #scale_fill_grey(start = 0.3)

ggsave('figures/capacity_cost.png')


### Comparison of total cost for other scenarios
sims %>%
   filter(Scenario %in% c('Baseline', 'Stoch. Cust.', 'Long Route'),
          Metric == 'Total Cost',
          Customers != 5) %>%
   mutate(Customers = factor(Customers)) %>%
   group_by(Scenario, Customers, Strategy, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Scenario, color = Scenario,
       linetype = Scenario, shape = Scenario) +
   geom_line(size = 1) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   facet_wrap(Strategy ~.) +
   scale_color_brewer(palette = "Dark2")
   #scale_color_grey(start = 0.3)

ggsave('figures/scenario_cost.png')


# Comparison of scenarios' distributions within individual strategies
strategies = unique(sims$Strategy)
for (strat in strategies){
   
   sims %>%
      filter(Scenario %in% c('Baseline', 'Short Route', 'Small Overlap', 'Stoch. Cust.'),
             Customers %in% c(10,80),
             Strategy == strat,
             Metric == 'Total Cost') %>%
      mutate(`Number of Customers` = factor(Customers)) %>%
      ggplot() +
      aes(x = Value, fill=`Number of Customers`) +
      geom_histogram(color='black', bins = 100) + 
      facet_grid(Scenario~.) +
      labs(x = 'Total Cost', y = 'Count') + 
      scale_fill_brewer(palette = "Dark2") +
      theme(aspect.ratio = 0.25, legend.position = 'top')
      #scale_fill_grey(start = 0.4)
   
   ggsave(paste('figures/hist_compare_',strat,'.png', sep = ''))
   
}



