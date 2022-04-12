
######################
### MEWS Output Analysis
######################

# Author: jpcarvallo

library(ggplot2)
library(tidyverse)
library(sqldf)
options(sqldf.driver = "SQLite")

# WD
setwd("C:\\Users\\jpcarvallo\\Documents\\CloudStation\\aa Reliability\\2021 ERMA\\work\\Task 4\\MEWS")

##########
# Read EPW data
##########

# Now in loop (IN USE)
fnames <- list.files('.\\epw_all_def\\sel\\')
epw_data_agg <- subset(test_filt, 1 == 2)

for (i in 1:length(fnames))
{
  test <- read.csv(paste0('.\\epw_all_def\\sel\\',fnames[i]),
                   header = F,
                   skip = 8)
  
  # I really only care about V1-V4 (year, month, day, hour) and then V7 (dry bulb temp)
  test_filt <- test[,c(1:4,7)]
  colnames(test_filt) <- c('year','month','day','he','dry_bulb_temp')
  test_filt$instance <- i
  epw_data_agg <- rbind(epw_data_agg, test_filt)
}

# Convert the non 2020 and 2060 in 9999, which will act as reference year
epw_data_agg$year_adj <- ifelse(epw_data_agg$year >= 2020, epw_data_agg$year, 9999)

# Summary by year
tapply(epw_data_agg$dry_bulb_temp, epw_data_agg$year_adj, summary)

# Density plots
ggplot(epw_data_agg, aes(x=dry_bulb_temp, color=factor(year_adj))) +
  geom_density() +
  #facet_wrap(~month) +
  theme_minimal()
  
ggplot(epw_data_agg, aes(x=factor(month), y=dry_bulb_temp, color=factor(year))) +
  geom_boxplot() +
  scale_color_discrete(name = 'Year') +
  #facet_wrap(~month) +
  theme_minimal() +
  labs(x='Month of year', y='Dry bulb temperature (C)')

##########
# Heat wave detection
##########

# Fist, p85 for monthly temp
p85_month <- epw_data_agg[epw_data_agg$year_adj == 9999,] %>%
  group_by(month) %>%
  summarize(quant50 = quantile(dry_bulb_temp, probs = 0.5),
            quant85 = quantile(dry_bulb_temp, probs = 0.85),
            quant90 = quantile(dry_bulb_temp, probs = 0.90))

# Adjustment to obtain quantiles from daily mins not from all hours
p85_month_2 <- epw_data_agg[epw_data_agg$year_adj == 9999,] %>%
  group_by(month, day) %>%
  summarize(daily_min = min(dry_bulb_temp)) %>%
  summarize(quant50 = quantile(daily_min, probs = 0.5),
            quant85 = quantile(daily_min, probs = 0.85),
            quant90 = quantile(daily_min, probs = 0.90))

# Then, detect days with min temp over p85 (note p90 for paper)
daily_temp <- sqldf("select year_adj, instance, month, day, 
                    min(dry_bulb_temp) as min_temp, avg(dry_bulb_temp) as avg_temp
                    from epw_data_agg
                    group by 1,2,3,4")

daily_temp_exp <- sqldf("select t1.*, 
                        CASE WHEN min_temp >= quant90 THEN 1 ELSE 0 END as det_min,
                        CASE WHEN avg_temp >= quant90 THEN 1 ELSE 0 END as det_avg
                        from daily_temp t1
                        join p85_month_2 using (month)")

with(daily_temp_exp, table(year_adj, det_min))
with(daily_temp_exp, table(year_adj, det_avg))

# Now detect consecutive days
get_streaks <- function(vec){
  x <- data.frame(trials=vec)
  x <- x %>% mutate(lagged=lag(trials)) %>% #note: that's dplyr::lag, not stats::lag
    mutate(start=(trials != lagged))
  x[1, "start"] <- TRUE
  x <- x %>% mutate(streak_id=cumsum(start))
  x <- x %>% group_by(streak_id) %>% mutate(streak=row_number()) %>%
    ungroup()
  return(x)
}

# First 2020
temp <- get_streaks(daily_temp_exp[daily_temp_exp$year_adj == 2020,'det_min'])
streaks_20 <- sqldf("select streak_id, max(streak) as hw_length from temp where trials = 1 group by 1")
streaks_20$year_adj <- 2020
# Find the daily temp (min/avg) for the HW days
temp2 <- cbind(temp, daily_temp_exp[daily_temp_exp$year_adj == 2020,])
temp2 <- temp2[temp2$trials == 1, c('streak_id','instance','month','day','min_temp','avg_temp')]
temp2 <- sqldf("select streak_id, instance, month, day, 
               min_temp - quant90 as exceed_min,
               avg_temp - quant90 as exceed_avg
               from temp2
               join p85_month_2 using (month)")
exced_20 <- sqldf("select streak_id, count(*) as ct, avg(exceed_min) as avg_ex_min, avg(exceed_avg) as avg_ex_avg
                  from temp2
                  group by 1")
exced_20 <- exced_20[exced_20$ct > 1,]
exced_20 %>% group_by(ct) %>% summarise(avg = max(avg_ex_min))

ct_hw_month_2020 <- sqldf("select month, count(*) as hw 
                          from (select streak_id, month, count(*) as dur from temp2 group by 1,2)
                          where dur > 1
                          group by 1")

ct_hw_month_2020$perc <- ct_hw_month_2020$hw/sum(ct_hw_month_2020$hw)

# Save the hourly profiles for these HW
hourly_hw_2020 <- sqldf("select year_adj, streak_id, instance, month, day, he, dry_bulb_temp
                        from epw_data_agg
                        join temp2 using (instance, month, day)
                        where year_adj = 2020")

# Now 2060
temp <- get_streaks(daily_temp_exp[daily_temp_exp$year_adj == 2060,'det_min'])
streaks_60 <- sqldf("select streak_id, max(streak) as hw_length from temp where trials = 1 group by 1")
streaks_60$year_adj <- 2060
# Find the daily temp (min/avg) for the HW days
temp2 <- cbind(temp, daily_temp_exp[daily_temp_exp$year_adj == 2060,])
temp2 <- temp2[temp2$trials == 1, c('streak_id','instance','month','day','min_temp','avg_temp')]
temp2 <- sqldf("select streak_id, instance, month, day, 
               min_temp - quant90 as exceed_min,
               avg_temp - quant90 as exceed_avg
               from temp2
               join p85_month_2 using (month)")
exced_60 <- sqldf("select streak_id, count(*) as ct, avg(exceed_min) as avg_ex_min, avg(exceed_avg) as avg_ex_avg
                  from temp2
                  group by 1")
exced_60 <- exced_60[exced_60$ct > 1,]
exced_60 %>% group_by(ct) %>% summarise(avg = max(avg_ex_min))

ct_hw_month_2060 <- sqldf("select month, count(*) as hw 
                          from (select streak_id, month, count(*) as dur from temp2 group by 1,2)
                          where dur > 1
                          group by 1")

ct_hw_month_2060$perc <- ct_hw_month_2060$hw/sum(ct_hw_month_2060$hw)

# Save the hourly profiles for these HW
hourly_hw_2060 <- sqldf("select year_adj, streak_id, instance, month, day, he, dry_bulb_temp
                        from epw_data_agg
                        join temp2 using (instance, month, day)
                        where year_adj = 2060")

write.csv(rbind(hourly_hw_2020, hourly_hw_2060),'hourly_hw_data.csv', row.names = F)

# Join streaks for plotting
streaks <- rbind(streaks_20, streaks_60)

ggplot(streaks[streaks$hw_length > 1,], aes(x=hw_length)) +
  geom_histogram(stat = 'count') +
  scale_x_continuous(labels = c(seq(2,10),15,20,25), breaks = c(seq(2,10),15,20,25)) +
  facet_wrap(~case_when(year_adj == 2020 ~ 'Year 2020',
                        year_adj == 2060 ~ 'Year 2060')) +
  labs(x='Heat wave length (days)', y='Number of heat waves in 100 instances') +
  theme_minimal(base_size = 20) +
  theme(axis.text.x = element_text(size = 18),
        text = element_text(family = "serif"))

ggsave("hw_hist_2020_2060.png", width = 11, height = 7, dpi = 1000)

# Alt fig with color instead of facet
ggplot(streaks[streaks$hw_length > 1,], aes(x=hw_length, 
                                            fill=case_when(year_adj == 2020 ~ 'Year 2020',
                                                           year_adj == 2060 ~ 'Year 2060'))) +
  geom_histogram(stat = 'count', position = 'dodge') +
  scale_x_continuous(labels = c(seq(2,10,2),15,20,25), breaks = c(seq(2,10,2),15,20,25)) +
  scale_fill_manual(name = '', values = c('red','blue')) +
  #facet_wrap(~case_when(year_adj == 2020 ~ 'Year 2020',
  #                      year_adj == 2060 ~ 'Year 2060')) +
  labs(x='Heat wave length (days)', y='Number of heat waves in 100 instances') +
  theme_minimal(base_size = 28) +
  theme(axis.text.x = element_text(size = 24),
        text = element_text(family = "serif"),
        legend.position = c(0.8,0.8))

ggsave("hw_hist_2020_2060_alt.png", width = 11, height = 7, dpi = 1000)

# Construct 8760 HW for visuals - 2020
hourly_hw_2020_8760 <- sqldf("select t2.year_adj, t1.month, t1.day, t1.he, avg(t2.dry_bulb_temp) as temp
                             from (select month, day, he from epw_data_agg where year_adj = 9999) t1
                             left join hourly_hw_2020 t2 using (month, day, he)
                             group by 1,2,3,4")

with(hourly_hw_2020, table(month, day)/24)

hourly_hw_2020_8760$hoy <- seq(1,8760)
hourly_hw_2020_8760$year_adj <- 2020
hourly_hw_2020_8760[is.na(hourly_hw_2020_8760$temp),'temp'] <- 0

ggplot(hourly_hw_2020_8760, aes(x=hoy, y=temp)) + geom_line() + coord_cartesian(xlim = c(8000,8900))

# Version for 1 instance. Choose 52 (instance with the most HW)
hourly_hw_2020_8760_i <- sqldf("select t2.year_adj, t1.month, t1.day, t1.he, t2.dry_bulb_temp as temp
                             from (select month, day, he from epw_data_agg where year_adj = 9999) t1
                             left join (select * from hourly_hw_2020 where instance = 52) t2
                             using (month, day, he)")

hourly_hw_2020_8760_i[is.na(hourly_hw_2020_8760_i$temp),'temp'] <- 0

# Counts of HW by month
sqldf("select month, count(*) from hourly_hw_2020_8760_i where temp > 0 group by 1")

# Construct 8760 HW for visuals - 2060
hourly_hw_2060_8760 <- sqldf("select t2.year_adj, t1.month, t1.day, t1.he, avg(t2.dry_bulb_temp) as temp
                             from (select month, day, he from epw_data_agg where year_adj = 9999) t1
                             left join hourly_hw_2060 t2 using (month, day, he)
                             group by 1,2,3,4
                             order by 2,3,4")

with(hourly_hw_2060, table(month, day)/24)

hourly_hw_2060_8760$hoy <- seq(1,8760)
hourly_hw_2060_8760$year_adj <- 2060
hourly_hw_2060_8760[is.na(hourly_hw_2060_8760$temp),'temp'] <- 0

hourly_hw_2060_8760$hoy_f <- factor(hourly_hw_2060_8760$hoy,
                                    levels = seq(1:8760))

ggplot(hourly_hw_2060_8760, aes(x=hoy, y=temp)) + geom_line() + coord_cartesian(xlim = c(3000,3150))

hourly_hw_2020_8760[hourly_hw_2020_8760$month == 5 & hourly_hw_2020_8760$day == 11,]


#########
# Older legacy code not in use

# Read an epw. Doesn't work
# test <- read_epw('.\\2060\\USA_NM_Albuquerque.Intl.AP.723650_TMY3SSP5-8.5_2060_r0.epw')

# Didn't work, try a csv approach
header <- c('Date','HH:MM','Datasource','DryBulb {C}','DewPoint {C}','RelHum {%}','Atmos Pressure {Pa}',
            'ExtHorzRad {Wh/m2}','ExtDirRad {Wh/m2}','HorzIRSky {Wh/m2}','GloHorzRad {Wh/m2}','DirNormRad {Wh/m2}',
            'DifHorzRad {Wh/m2}','GloHorzIllum {lux}','DirNormIllum {lux}','DifHorzIllum {lux}','ZenLum {Cd/m2}','WindDir {deg}',
            'WindSpd {m/s}','TotSkyCvr {.1}','OpaqSkyCvr {.1}','Visibility {km}','Ceiling Hgt {m}','PresWeathObs','PresWeathCodes',
            'Precip Wtr {mm}','Aerosol Opt Depth {.001}','SnowDepth {cm}','Days Last Snow','Albedo {.01}','Rain {mm}','Rain Quantity {hr}')

test <- read.csv('.\\2060\\USA_NM_Albuquerque.Intl.AP.723650_TMY3SSP5-8.5_2060_r0.epw',
                 header = F,
                 skip = 8)

# I really only care about V1-V4 (year, month, day, hour) and then V7 (dry bulb temp)
test_filt <- test[,c(1:4,7)]
colnames(test_filt) <- c('year','month','day','he','dry_bulb_temp')
test_filt$instance <- 1
