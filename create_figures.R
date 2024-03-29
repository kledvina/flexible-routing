# create_figures.R
# Reads in all simulation output files from 'output/' folder
# Creates and exports summary figures

#---------- Setup ----------#

library(tidyverse)
library(readxl)
library(RColorBrewer)
theme_set(theme_bw())
#theme_set(theme_bw(base_size = 22))
theme_set(theme_bw(base_size = 11))

setwd("~/Documents/flexible-routing") # Set to file directory
outpath <- "C:/Users/hanzh/Documents/GitHub/flexible-routing/figures/" # Set to relative location of folder for figures
num_sims <- 6000 # As set in Python simulation code


#---------- Dataset Preparation ----------#

# Get simulation output files
#files <- list.files('output', pattern = '.xlsx')

# Import and combine datasets with individual runs
#sims <- data.frame()
#for (i  in 1:length(files)){
#   df <- read_xlsx(paste('output/', files[i], sep = ''), sheet = 1)
#   sims <- rbind(sims, df)
#}

sims = read_xlsx('C:/Users/hanzh/Documents/GitHub/flexible-routing/output/results_2021-08-31_16-24-47.xlsx', sheet=1)
colnames(sims) = c('ID','Scenario','Customers','Strategy','Metric','Value')

# Rename scenarios
sims[(sims$Scenario == 'baseline'),]$Scenario = 'Baseline'
#sims[(sims$Scenario == 'baseline_k3'),]$Scenario = 'Medium Overlap'
#sims[(sims$Scenario == 'baseline_k1'),]$Scenario = 'Small Overlap'
#sims[(sims$Scenario == 'short_route'),]$Scenario = 'Short Route'
#sims[(sims$Scenario == 'long_route'),]$Scenario = 'Long Route'
#sims[(sims$Scenario == 'stochastic_customers'),]$Scenario = 'Stoch. Cust.'
#sims[(sims$Scenario == 'binomial'),]$Scenario = 'Bin. Demand'
#sims[(sims$Scenario == 'low_capacity'),]$Scenario = 'Low Capacity'
#sims[(sims$Scenario == 'high_capacity'),]$Scenario = 'High Capacity'

# Rename and reorder routing strategies
sims[(sims$Strategy == 'dedicated'),]$Strategy = 'Dedicated'
sims[(sims$Strategy == 'overlapped'),]$Strategy = 'AO'
sims[(sims$Strategy == 'overlapped closed'),]$Strategy = 'RAO'
sims[(sims$Strategy == 'reoptimization'),]$Strategy = 'Reoptimization'
sims[(sims$Strategy == 'fully flexible'),]$Strategy = 'FO'
sims[(sims$Strategy == 'fully flexible closed'),]$Strategy = 'RFO'

sims$Strategy = factor(sims$Strategy,
                       levels = c('Dedicated', 'AO', 'FO', 'RAO', 'RFO', 'Reoptimization'))

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


#---------- Baseline Graphs (Open Chain) ----------#
# Total cost
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Total Cost',
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 0.5) + geom_point(size = 2) +
   expand_limits(y=0) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   theme(aspect.ratio = 0.75)
ggsave(paste(outpath, 'total_cost.png', sep=''))


# Relative to Reoptimization
reopt <- sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy == 'Reoptimization') %>%
   group_by(Customers) %>%
   summarise(avg_reopt = mean(Value))

sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   merge(reopt) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)/mean(avg_reopt)) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Strategy, 
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 0.5) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost (Rel. to Reoptimization)') + 
   expand_limits(y=0) +
   scale_color_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.75)
ggsave(paste(outpath, 'rel_cost.png', sep=''))

# Combined: Total & Relative to Reoptimization
sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   merge(reopt) %>%
   merge(reopt) %>%
   group_by(Customers, Strategy) %>%
   summarise(`Total Cost` = mean(Value),
             `Relative Cost` = mean(Value)/mean(avg_reopt)) %>%
   gather(Metric, Value, `Total Cost`, `Relative Cost`) %>%
   mutate(Metric = factor(Metric, levels = c("Total Cost", "Relative Cost"))) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 0.5) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   expand_limits(y=0) +
   facet_wrap(Metric ~., scales = 'free') +
   theme(aspect.ratio = 1.25)
ggsave(paste(outpath, 'combined_total_rel_cost.png', sep=''))


# Circular and Radial Cost
sims %>%
   filter(Scenario == 'Baseline',
          Metric %in% c('Circular Cost', 'Radial Cost'),
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   group_by(Customers, Strategy, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Strategy,
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 0.5) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   facet_wrap(Metric ~., nrow=1) +
   theme(aspect.ratio = 2) +
   scale_fill_grey(start = 0.4) +
   theme(aspect.ratio = 1.25)
ggsave(paste(outpath, 'cost_breakdown.png', sep=''))

sims %>%
   filter(Scenario == 'Baseline',
          Metric %in% c('Circular Cost', 'Radial Cost'),
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   group_by(Customers, Strategy, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Metric, fill = Metric) +
   geom_area() +
   labs(x = 'Number of Customers', y = 'Cost') + 
   facet_wrap(Strategy ~., nrow=1) +
   theme(aspect.ratio = 2, legend.position = 'bottom', legend.title = element_blank()) +
   scale_fill_grey(start = 0.4)
ggsave(paste(outpath, 'cost_breakdown_area.png', sep=''))


# Number of trips
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Trip Count',
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   mutate(Customers = factor(Customers)) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y = Value, fill = Strategy) + 
   geom_bar(stat = 'identity', position = 'dodge') +
   labs(x = 'Number of Customers', y = 'Trip Count') + 
   theme(aspect.ratio = 0.75) +
   scale_fill_grey(start = 0.3)

#ggsave(paste(outpath, 'trips.png', sep=''))


# Distribution of individual runs' costs by strategy
sims %>%
   filter(Scenario == 'Baseline',
          Customers %in% c(5,20,80),
          Metric == 'Total Cost') %>%
   group_by(Customers, Strategy) %>%
   summarise(mean(Value),
             median(Value),
             sd(Value))

sims %>%
   filter(Scenario == 'Baseline',
          Customers %in% c(5,20,80),
          Metric == 'Total Cost',
          Strategy %in% c('Dedicated','AO','FO','Reoptimization')) %>%
   mutate(`Number of Customers` = factor(Customers)) %>%
   ggplot() +
   aes(x = Value, fill=`Number of Customers`) +
   geom_histogram(color='black', bins = 50) + 
   facet_grid(Strategy~.) +
   labs(x = 'Total Cost', y = 'Count') + 
   theme(aspect.ratio = 0.25, legend.position="top") +
   scale_fill_grey(start = 0.3, end = 0.9)

ggsave(paste(outpath, 'hist_total.png', sep=''))


# Percent of sims where AO did better than dedicated
sims %>%
   filter(Scenario == 'Baseline', Metric == 'Total Cost') %>%
   spread(Strategy, Value) %>%
   group_by(Scenario, Customers) %>%
   summarise(`Lower Cost` = 100*sum(AO < Dedicated) / num_sims,
             `Equal Cost` = 100*sum(AO == Dedicated) / num_sims,
             `Higher Cost` = 100*sum(AO > Dedicated) / num_sims)




#---------- Baseline Graphs (Closed Chain) ----------#

# Total cost
sims %>%
   filter(Scenario == 'Baseline',
          Metric == 'Total Cost',
          Strategy %in% c('Dedicated','AO','RAO','FO','RFO','Reoptimization')) %>%
   mutate(Customers= factor(Customers)) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value, fill = Strategy) +
   geom_bar(stat = 'identity', position = 'dodge') +
   expand_limits(y=0) +
   labs(x = 'Number of Customers', y = 'Cost') + 
   theme(aspect.ratio = 1) +
   scale_fill_grey(start = 0.3)
ggsave(paste(outpath, 'total_cost_CLOSED.png', sep=''))



# Relative to Reoptimization -- LINE
reopt <- sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy == 'Reoptimization') %>%
   group_by(Customers) %>%
   summarise(avg_reopt = mean(Value))

sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy %in% c('AO','FO','RAO','RFO')) %>%
   merge(reopt) %>%
   mutate(Customers= factor(Customers)) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)/mean(avg_reopt) - 1) %>%
   ggplot() +
   aes(x = Customers, y =  Value, group = Strategy, 
       linetype = Strategy, shape = Strategy) +
   geom_line(size = 0.5) + geom_point(size = 2) +
   labs(x = 'Number of Customers', y = 'Cost (% Above Reoptimization)') + 
   expand_limits(y=0) +
   theme(aspect.ratio = 0.75) +
   scale_y_continuous(labels = scales::percent) +
   scale_fill_grey(start = 0.3)
ggsave(paste(outpath, 'rel_cost_CLOSED_line.png', sep=''))

# Relative to Reoptimization -- BAR
reopt <- sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy == 'Reoptimization') %>%
   group_by(Customers) %>%
   summarise(avg_reopt = mean(Value))

sims %>%
   filter(Scenario == "Baseline",
          Metric == 'Total Cost',
          Strategy %in% c('AO','FO','RAO','RFO')) %>%
   merge(reopt) %>%
   mutate(Customers= factor(Customers)) %>%
   group_by(Customers, Strategy) %>%
   summarise(Value = mean(Value)/mean(avg_reopt) - 1) %>%
   ggplot() +
   aes(x = Customers, y =  Value, fill = Strategy) +
   geom_bar(stat = 'identity', position = 'dodge') +
   labs(x = 'Number of Customers', y = 'Cost (% Above Reoptimization)') + 
   expand_limits(y=0) +
   theme(aspect.ratio = 0.75) +
   scale_y_continuous(labels = scales::percent) +
   scale_fill_grey(start = 0.3)
ggsave(paste(outpath, 'rel_cost_CLOSED_bar.png', sep=''))




###################### OLD (BEFORE 2021) #############################

#---------- Scenario Comparisons ----------#

#--- Overlap size ---#

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
   theme(aspect.ratio = 1.25) +
   scale_fill_brewer(palette = "Dark2")
   #scale_fill_grey(start = 0.3)

ggsave(paste(outpath, 'overlap_size_cost.png', sep=''))


#---Capacity---#

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
   geom_hline(yintercept = 1.0, alpha = 0.75, linetype = 'dashed') +
   labs(x = 'Number of Customers', y = 'Cost (Rel. to Baseline)') + 
   facet_wrap(Strategy ~.) +
   theme(aspect.ratio = 1.25) +
   scale_fill_brewer(palette = "Dark2")
   #scale_fill_grey(start = 0.3)

ggsave(paste(outpath, 'capacity_cost.png', sep=''))


#---Route size---#

# Circular and radial cost breakdown
sims %>%
   filter(Scenario %in% c('Baseline', 'Long Route', 'Short Route'),
          Metric %in% c('Circular Cost', 'Radial Cost'),
          !Customers %in% c(4,5)) %>%
   group_by(Customers, Strategy, Scenario, Metric) %>%
   summarise(Value = mean(Value)) %>%
   ggplot() +
   aes(x = Customers, y =  Value,
       group = Metric, fill = Metric) +
   geom_area() +
   labs(x = 'Number of Customers', y = 'Cost') + 
   facet_grid(Strategy ~ Scenario) +
   scale_fill_brewer(palette = "Dark2") +
   expand_limits(x = 0, y = 0) +
   theme(aspect.ratio = 0.75)
   #scale_fill_grey(start = 0.4)

#ggsave(paste(outpath, 'route_size_cost_breakdown.png', sep=''))


# Combined: Total & Relative to Reoptimization
reopt_routesize <- sims %>%
   filter(Scenario %in% c('Baseline', 'Long Route', 'Short Route'),
          Metric == 'Total Cost',
          Strategy == 'Reoptimization',
          !Customers %in% c(4,5)) %>%
   group_by(Customers, Scenario) %>%
   summarise(avg_reopt = mean(Value))

sims %>%
   filter(Scenario %in% c('Baseline', 'Long Route', 'Short Route'),
          Metric == 'Total Cost',
          !Customers %in% c(4,5)) %>%
   merge(reopt_routesize) %>%
   group_by(Customers, Strategy, Scenario) %>%
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
   facet_grid(Metric ~ Scenario, scales = 'free') +
   scale_color_brewer(palette = "Dark2") +
   expand_limits(x = 0, y = 0) +
   theme(aspect.ratio = 1.25)

ggsave(paste(outpath, 'route_size_cost.png', sep=''))



#--- Demand distributions ---#

# Percent of sims where overlapped did better than dedicated
share_table <- sims %>%
   filter(Scenario %in% c('Baseline', 'Bin. Demand', 'Stoch. Cust.'),
          Metric == 'Total Cost') %>%
   spread(Strategy, Value) %>%
   group_by(Scenario, Customers) %>%
   summarise(`Lower Cost` = 100*sum(Overlapped < Dedicated) / num_sims,
             `Equal Cost` = 100*sum(Overlapped == Dedicated) / num_sims,
             `Higher Cost` = 100*sum(Overlapped > Dedicated) / num_sims)

share_table

share_table %>%
   gather(`Overlapped Cost`, Percent, `Lower Cost`, `Equal Cost`, `Higher Cost`) %>%
   #mutate(Customers = factor(Customers)) %>%
   ggplot() +
   aes(x = Customers, y = Percent, color = Scenario, group = Scenario,
       linetype = Scenario, shape = Scenario) +
   geom_line(size = 1) + geom_point(size = 2) +
   labs(x = 'Number of Customers') +
   facet_wrap(`Overlapped Cost` ~.) +
   scale_color_brewer(palette = "Dark2") +
   expand_limits(x = 0, y = 0) +
   theme(aspect.ratio = 1.25)

ggsave(paste(outpath, 'dem_scen_cost.png', sep=''))


# Comparison of scenarios' distributions within individual strategies
sims %>%
   filter(Scenario %in% c('Baseline', 'Bin. Demand', 'Stoch. Cust.'),
          Customers %in% c(10,80),
          Metric == 'Total Cost') %>%
   mutate(`Number of Customers` = factor(Customers)) %>%
   ggplot() +
   aes(x = Value, fill=`Number of Customers`) +
   geom_histogram(color='black', bins = 40) + 
   facet_grid(Scenario~Strategy) +
   labs(x = 'Total Cost', y = 'Count') + 
   scale_fill_brewer(palette = "Dark2") +
   theme(aspect.ratio = 0.5, legend.position = 'top')
#scale_fill_grey(start = 0.4)

ggsave(paste(outpath, 'dem_scen_hists.png', sep=''))


