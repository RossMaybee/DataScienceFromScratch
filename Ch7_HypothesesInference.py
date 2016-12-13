# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 11:33:45 2016

@author: ross

Data Science From Scratch
Chapter 7: Hypothesis & Inference
"""

from __future__ import division
from collections import Counter
import math, random
import matplotlib.pyplot as plt


    # Including prior functions from Ch 6
def normal_pdf(x, mu=0, sigma=1):
    sqrt_two_pi = math.sqrt(2 * math.pi)
    return (math.exp(-(x-mu) ** 2 / 2 / sigma **2) / (sqrt_two_pi * sigma))
"""    
REMOVING THE CH6 PLOTS FROM OUR CH7 WORK
xs = [x / 10.0 for x in range (-50,50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma-1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
plt.show()
"""
def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x-mu) / math.sqrt(2) / sigma)) / 2
"""
REMOVING THE CH6 PLOTS FROM OUR CH7 WORK    
xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')
plt.legend(loc=4) #bottom right
plt.title("Various normal cdfs")
plt.show()
"""

def inverse_normal_cdf(p, mu=0, sigma=1, tolerance=0.00001):
    """find approximate inverse using binary search"""
    
    # if not standard, compute standards and rescale
    if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
        
    low_z, low_p = -10.0, 0     # normal_cdf(-10) is (very close to ) 0
    hi_z, hi_p = 10.0, 1        # normal_cdf(10) is (very close to) 1
    while hi_z - low_z > tolerance:
            mid_z = (low_z + hi_z) / 2  #consider the midpoint
            mid_p = normal_cdf(mid_z)   #and the cdf's value there
            if mid_p < p:       # midpoint is still too low, search above it
                low_z, low_p = mid_z, mid_p
            elif mid_p > p:     # midpoint is still too high, search below it
                hi_z, hi_p = mid_z, mid_p
            else:
                break
    return mid_z

# Now the Ch 7 work begins

def normal_approximation_to_binomial(n, p):
    """finds mu and sigma corresponding to a Binomial(n,p)"""
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma
    
# the normal cdf _is_ the probability the variable is below a threshold
normal_probability_below = normal_cdf

# it's above the threshold if it's not below the threshold
def normal_probability_above(lo, mu=0, sigma=1):
    return 1 - normal_cdf(lo, mu, sigma)

# it's between if it's less than hi, but not less than lo
def normal_probability_between(lo, hi, mu=0, sigma=1):
    return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# it's outside if it's not between
def normal_probability_outside(lo, hi, mu=0, sigma=1):
    return 1 - normal_probability_between(lo, hi, mu, sigma)

def normal_upper_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z <= z) = probability"""
    return inverse_normal_cdf(probability, mu, sigma)
    
def normal_lower_bound(probability, mu=0, sigma=1):
    """returns the z for which P(Z >= z) = probability"""
    return inverse_normal_cdf(1 - probability, mu, sigma)
    
def normal_two_sided_bounds(probability, mu=0, sigma=1):
    """returns the symmetric (about the mean) bounds that contain
    the specified probability"""
    tail_probability = (1 - probability) / 2
    
    #upper bound should have tail_probability above it
    upper_bound = normal_lower_bound(tail_probability, mu, sigma)
    
    #lower bound should have tail_probability below it
    lower_bound = normal_upper_bound(tail_probability, mu, sigma)
    
    return lower_bound, upper_bound
    
mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

normal_two_sided_bounds(0.95, mu_0, sigma_0) #(469, 531)

# 95% bounds based on assumption p is 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# actual mu and sigma based on p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# a type 2 error means we fail to reject the null hypothesis
# which will happen when X is still in our original interval

type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability
# 0.887

hi = normal_upper_bound(0.95, mu_0, sigma_0)
# is 526 (<531, since we need more probability in the upper tail)

type_2_probability = normal_probability_below(hi, mu_1, sigma_1)

power = 1 - type_2_probability
# 0.936

def two_sided_p_value(x, mu=0, sigma=1):
    if x >= mu:
        #if x is greater than the mean, the tail is what's greater than x
        return 2 * normal_probability_above(x, mu, sigma)
    else:
        #if x is less than the mean, the tail is what's less than x
        return 2 * normal_probability_below(x, mu, sigma)
        
two_sided_p_value(529.5, mu_0, sigma_0)
# 529.5 for *continuity correction*

extreme_value_count = 0
for _ in range(100000):
    num_heads = sum(1 if random.random() < 0.5 else 0   # count # of heads
        for _ in range(1000))                           # in 10000 flips
    if num_heads >= 530 or num_heads <= 470:            # and count how often
        extreme_value_count += 1                            # the # is extreme
    
print extreme_value_count / 100000                      # 0.062
                                
two_sided_p_value(531.5, mu_0, sigma_0)                 # 0.0463

upper_p_value = normal_probability_above
lower_p_value = normal_probability_below

upper_p_value(524.5, mu_0, sigma_0)                     # 0.061

upper_p_value(526.5, mu_0, sigma_0)                     # 0.047

p_hat = 525 / 1000
mu = p_hat
sigma = math.sqrt(p_hat * (1 - p_hat) / 1000)           # 0.0158

normal_two_sided_bounds(0.95, mu, sigma)                #[0.494, 0.556]

def run_experiment():
    """flip a fair coin 1000 times, True = heads, False = tails"""
    return [random.random() < 0.5 for _ in range(1000)]
    
def reject_fairness(experiment):
    """using the 5% significance levels"""
    num_heads = len([flip for flip in experiment if flip])
    return num_heads < 469 or num_heads > 531

random.seed(0)
experiments = [run_experiment() for _ in range(1000)]
num_rejections = len([experiment for experiment in experiments
    if reject_fairness(experiment)])

print num_rejections #46        


def estimated_parameters(N, n):
    p = n / N
    sigma = math.sqrt(p * (1 - p) / N)
    return p, sigma
    
def a_b_test_statistic(N_A, n_A, N_B, n_B):
    p_A, sigma_A = estimated_parameters(N_A, n_A)
    p_B, sigma_B = estimated_parameters(N_B, n_B)
    return (p_B - p_A) / math.sqrt(sigma_A ** 2 + sigma_B ** 2)

z = a_b_test_statistic(1000, 200, 1000, 180)    # - 1.14

two_sided_p_value(z)                            # 0.254

z = a_b_test_statistic(1000, 200, 1000, 150)    # -2.94

two_sided_p_value(z)                            # 0.003

def B(alpha, beta):
    """a normalizing constant so that the total probability is 1"""
    return math.gamma(alpha) * math.gamma(beta) / math.gamma(alpha + beta)

def beta_pdf(x, alpha, beta):
    if x < 0 or x > 1:          # no weight outside of [0, 1]
        return 0
    return x ** (alpha - 1) * (1 - x) ** (beta - 1) / B(alpha, beta)
