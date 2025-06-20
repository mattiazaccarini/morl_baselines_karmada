import numpy as np
import gymnasium as gym
import math
import time
import heapq
from gymnasium import spaces
import pandas as pd
import wandb

from statistics import mean
from envs.utils import DeploymentRequest, get_c2e_deployment_list, calculate_gini_coefficient, sort_dict_by_value, save_to_csv_multi

MAX_REPLICAS = 8
MIN_REPLICAS = 1

DEFAULT_NUM_CLUSTERS = 4
DEFAULT_ARRIVAL_RATE = 100
DEFAULT_CALL_DURATION = 1

# Spreading strategies
NUM_SPREADING_ACTIONS = 1
FFD = 0  # First Fit Deployment

DEFAULT_FILE_NAME_RESULTS = "karmada_gym_results"
NUM_METRICS_CLUSTER = 4
NUM_METRICS_REQUEST = 4

DEFAULT_NODE_TYPE = "vWall"

# Cluster types
NUM_CLUSTER_TYPES = 5  # edge_tier_1, edge_tier_2, fog_tier_1, fog_tier_2, cloud

OLD_CLUSTER_TYPES = [{"type": "edge_tier_1", "cpu": 2.0, "mem": 2.0, "cost": 1},
                         {"type": "edge_tier_2", "cpu": 2.0, "mem": 4.0, "cost": 2},
                         {"type": "fog_tier_1", "cpu": 2.0, "mem": 8.0, "cost": 4},
                         {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 8},
                         {"type": "cloud", "cpu": 8.0, "mem": 32.0, "cost": 16}]

# TODO: Check cost values for the clusters, which are greater than the one used for FGCS
# The reported values for CPU and Memory are the real ones, there is no much difference between the CPU values
# Many eddge devices have multiple cores now.

DEFAULT_CLUSTER_TYPES =  [{"type": "edge_tier_1", "cpu": 4.0, "mem": 1.0, "cost": 2.0, "device": "raspi3"}, # Raspi3
                         {"type": "edge_tier_2", "cpu": 4.0, "mem": 4.0, "cost": 4.0, "device": "raspi4"}, # Raspi4
                         {"type": "fog_tier_1", "cpu": 4.0, "mem": 16.0, "cost": 16.0, "device": "shuttle"}, # Shuttle
                         {"type": "fog_tier_2", "cpu": 4.0, "mem": 16.0, "cost": 32.0, "device": "intel_nuc"}, # Intel NUC
                         {"type": "cloud", "cpu": 8.0, "mem": 48.0, "cost": 64, "device": "vwall"}, # vwall
                         {"type": "cloud", "cpu": 8.0, "mem": 64.0, "cost": 72, "device": "ecluster"}] # eCluster

DEFAULT_NUM_EPISODE_STEPS = 250

# Defaults for latency
MIN_DELAY = 1  # corresponds to 1ms
MAX_DELAY = 1000  # corresponds to 1000ms

class KarmadaSchedulingEnvMultiPower(gym.Env):
    def __init__(self, num_clusters=DEFAULT_NUM_CLUSTERS,
                 arrival_rate_r=DEFAULT_ARRIVAL_RATE,
                 call_duration_r=DEFAULT_CALL_DURATION,
                 episode_length=DEFAULT_NUM_EPISODE_STEPS,
                 min_replicas=MIN_REPLICAS,
                 max_replicas=MAX_REPLICAS,
                 file_results_name=DEFAULT_FILE_NAME_RESULTS,
                 is_eval_env=False):
        # Define action and observation space
        super(KarmadaSchedulingEnvMultiPower, self).__init__()

        self.current_step = 0
        self.current_time = 0
        self.time_start = 0
        self.penalty = False
        self.episode_over = False
        self.episode_count = 0


        self.offered_requests = 0

        self.num_clusters = num_clusters
        self.arrival_rate_r = arrival_rate_r
        self.call_duration_r = call_duration_r
        self.episode_length = episode_length
        self.running_requests: list[DeploymentRequest] = []
        
        # Latency purposes
        self.latency = np.zeros(num_clusters, dtype=np.float32)  # Average latency per cluster
        self.latency_matrix = np.zeros((num_clusters, num_clusters))

        # For Request generation
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        # Action space --> Deploy service in a specific cluster + spreading actions + reject
        self.num_actions = num_clusters + NUM_SPREADING_ACTIONS #+ 1

        self.observation_space = spaces.Box(low=0.0,
                                            high=1000.0,
                                            shape=(num_clusters + NUM_SPREADING_ACTIONS + 1,
                                                   NUM_METRICS_CLUSTER + NUM_METRICS_REQUEST + 2),
                                            dtype=np.float32)
        
        
        self.action_space = spaces.Discrete(self.num_actions)

        self.reward_dim = 4  # Latency, Cost, Gini, Power Consumption
        low = np.array([-np.inf] * self.reward_dim, dtype=np.float32)
        high = np.array([np.inf] * self.reward_dim, dtype=np.float32)

        self.reward_space = spaces.Box(low=low, high=high, shape=(self.reward_dim,), dtype=np.float32)
        self.info = {}
        
        # Setting the experiment based on Cloud2Edge (C2E) deployments
        self.deploymentList = get_c2e_deployment_list()
        self.deployment_request = None

        # Cluster type initialization, resource capacities based on cluster types
        self.cpu_capacity = np.zeros(num_clusters, dtype=np.float32)
        self.memory_capacity = np.zeros(num_clusters, dtype=np.float32)
        # Initialize a list to keep track of the power consumption of each cluster
        self.cluster_power_consumption = np.zeros(num_clusters, dtype=np.float32)
        self.default_cluster_types = DEFAULT_CLUSTER_TYPES
        self.cluster_type = [0] * num_clusters  # Initialize with default cluster types
        
        for c in range(num_clusters):
            type = int(self.np_random.integers(low=0, high=NUM_CLUSTER_TYPES))
            self.cluster_type[c] = type
            self.cpu_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['cpu']
            self.memory_capacity[c] = DEFAULT_CLUSTER_TYPES[type]['mem']
        
        # Default latency matrix
        for n1 in range(num_clusters):
            for n2 in range(num_clusters):
                if n1 == n2:  # for the same node assume 0
                    self.latency_matrix[n1][n2] = 0
                else:
                    self.latency_matrix[n1][n2] = self.np_random.integers(low=MIN_DELAY, high=MAX_DELAY)

            self.latency[n1] = mean(self.latency_matrix[n1])

        self.default_cluster_types = DEFAULT_CLUSTER_TYPES
        # Found this, was it a bug? It overwrites the cluster types set before. 
        #self.cluster_types = [0] * num_clusters  # Initialize with default cluster types

        # Read the power consumption data from CSV file
        self.power_consumption_data = self.read_power_consumption()
        self.total_power_consumption = 0.0
        #Info and utilized for Gini coefficient calculation
        self.avg_load_served = np.zeros(num_clusters, dtype=np.float32)

        # Keep track of allocated resources
        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters).astype(np.float32)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=num_clusters).astype(np.float32)

        # This value is 0 at the start, and then is updated in the method direclty
        self.calculate_power_consumption()

        # Keep track of free resources for deployment requests
        self.free_cpu = np.zeros(num_clusters, dtype=np.float32)
        self.free_memory = np.zeros(num_clusters, dtype=np.float32)

        # Variables for spreading strategies
        self.split_number_replicas = np.zeros(num_clusters, dtype=np.int32)  # Number of replicas per cluster

        # Keep track of deployment actions
        self.deploy_ffd = 0  # First Fit Deployment

        self.file_results = file_results_name + ".csv"
        self.is_eval_env = is_eval_env
        self.accepted_requests = 0
        self.ep_accepted_requests = 0
        self.deploy_all = 0
        self.avg_cpu_usage_percentage_cluster_selected = []
        self.avg_latency = []
        self.avg_cost = []
        self.next_request()


    def reset(self, seed, **kwargs):
        # Reset the environment state and return the initial observation
        # print('Resetting environment...')
        # Try logging the power consumption to wandb
        if self.total_power_consumption and self.total_power_consumption > 0:
            # Log the total power consumption and the power consumption for each cluster
            wandb.log({
                "total_power_consumption": self.total_power_consumption,
                # log also the power consumption for each cluster
                **{f"cluster_{i}_power": v for i, v in enumerate(self.cluster_power_consumption)}
            })
        
        self.current_step = 0
        self.penalty = False
        self.episode_over = False
        self.avg_cpu_usage_percentage_cluster_selected = []

        for n1 in range(self.num_clusters):
            for n2 in range(self.num_clusters):
                if n1 == n2:  # for the same node assume 0
                    self.latency_matrix[n1][n2] = 0
                else:
                    self.latency_matrix[n1][n2] = self.np_random.integers(low=MIN_DELAY, high=MAX_DELAY)

            self.latency[n1] = mean(self.latency_matrix[n1])

        # Reset Deployment Data
        self.deploymentList = get_c2e_deployment_list()

        self.avg_load_served = np.zeros(self.num_clusters, dtype=np.float32)

        # Keep track of allocated resources
        self.allocated_cpu = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters).astype(np.float32)
        self.allocated_memory = self.np_random.uniform(low=0.0, high=0.2, size=self.num_clusters).astype(np.float32)

        self.total_power_consumption = self.calculate_power_consumption()

        # Variables for spreading strategies
        self.split_number_replicas = np.zeros(self.num_clusters, dtype=np.int32)  # Number of replicas per cluster

        # Keep track of spreading actions
        self.deploy_ffd = 0  # First Fit Deployment
        self.info = {}

        # return obs
        #print('Resetting environment...')
        #print(f'State: {np.array(self.get_state())}')
        #return np.array(self.get_state()), {}
        state = self.get_state()
        return tuple(state.flatten()), {}

    def step(self, action):
        if self.current_step == 1:
            self.time_start = time.time()

        self.offered_requests += 1
        
        # TODO: Keep track of the power consumption here and pass to get_reward() only the increment or decrement
        current_power_consumption = self.total_power_consumption
        self.take_action(action)
        # update the power consumption after taking the action
        self.calculate_power_consumption()
        #print(f"Power consumption changed from {current_power_consumption} to {self.total_power_consumption} after action {action}")

        reward = self.get_reward()

        # Get next request
        self.next_request()

        # Update observation
        ob = self.get_state()
        #print(f"State: {ob}")
        #print(f"Shape: {ob.shape}")
        #print(f"Type: {ob.dtype}")

        if self.current_step == self.episode_length:
            self.episode_count += 1
            self.episode_over = True
            self.execution_time = time.time() - self.time_start

            gini = calculate_gini_coefficient(self.avg_load_served)

            if self.is_eval_env:
                save_to_csv_multi(self.file_results, self.episode_count,
                    self.ep_accepted_requests,
                    self.episode_length - self.ep_accepted_requests,
                    float(np.mean(self.avg_latency)) if self.avg_latency else float(MAX_DELAY),
                    float(np.mean(self.avg_cost))    if self.avg_cost    else float(MAX_COST),
                    float(np.mean(self.avg_cpu_usage_percentage_cluster_selected)) if self.avg_cpu_usage_percentage_cluster_selected else 0.0,
                    gini,
                    self.execution_time)
            else:
                save_to_csv_multi(self.file_results, self.episode_count,
                    self.ep_accepted_requests,
                    self.episode_length - self.ep_accepted_requests,
                    float(np.mean(self.avg_latency)) if self.avg_latency else float(MAX_DELAY),
                    float(np.mean(self.avg_cost))    if self.avg_cost    else float(MAX_COST),
                    float(np.mean(self.avg_cpu_usage_percentage_cluster_selected)) if self.avg_cpu_usage_percentage_cluster_selected else 0.0,
                    gini,
                    self.execution_time)

        
        #return np.array(ob), reward, self.episode_over, False, self.info
        
        # Use these lines with MOQLearning
        obs = self.get_state()
        return tuple(obs.flatten()), reward, self.episode_over, False, {}

    # Apply the action taken by the agent
    def take_action(self, action):
        self.current_step += 1
        #print(f"Action taken: {action} at step {self.current_step}")
        # Stop if MAX_STEPS
        if self.current_step == self.episode_length:
            self.episode_over = True
        
        if action < self.num_clusters: # Deploy to a specific cluster
            if self.check_if_cluster_is_full_after_full_deployment(action):
                self.penalty = True
            else:
                # Accept request
                self.accepted_requests += 1
                self.ep_accepted_requests += 1
                self.deploy_all += 1
                self.deployment_request.deployed_cluster = action
                self.penalty = False
                # Update allocated resources
                self.allocated_cpu[action] += self.deployment_request.num_replicas * self.deployment_request.cpu_request
                self.allocated_memory[action] += self.deployment_request.num_replicas * self.deployment_request.memory_request

                self.avg_cpu_usage_percentage_cluster_selected.append(
                    100 * (self.allocated_cpu[action] / self.cpu_capacity[action]))
                self.avg_load_served[action] += self.deployment_request.num_replicas

                # Update free resources
                self.free_cpu[action] = self.cpu_capacity[action] - self.allocated_cpu[action]
                self.free_memory[action] = self.memory_capacity[action] - self.allocated_memory[action]
                self.enqueue_request(self.deployment_request)

                # Latency and cost updates
                self.increase_latency(action, 1.15) # Increase latency by 15% for the deployed cluster
                self.avg_latency.append(self.latency[action])
                type_id = self.cluster_type[action]
                self.avg_cost.append(DEFAULT_CLUSTER_TYPES[type_id]['cost'])

                # Save expected latency and cost for the deployment request
                self.deployment_request.expected_latency = self.latency[action]
                self.deployment_request.expected_cost = DEFAULT_CLUSTER_TYPES[type_id]['cost']
        elif action == self.num_clusters + FFD: # First Fit Deployment Spreading Strategy
            #print('Action taken: First Fit Deployment Spreading Strategy')
            if self.deployment_request.num_replicas == 1:
                self.penalty = True # Cannot spread a single replica
            else:
                div = self.first_fit_decreasing_heuristic(self.deployment_request.num_replicas,
                                                          self.deployment_request.cpu_request,
                                                          self.deployment_request.memory_request, self.num_clusters,
                                                          self.free_cpu, self.free_memory)
                if self.check_if_clusters_are_full_after_split_deployment(div):
                    self.penalty = True
                else:
                    # Accept request
                    self.penalty = False
                    self.accepted_requests += 1
                    self.ep_accepted_requests += 1
                    self.deploy_ffd += 1
                    self.deployment_request.split_clusters = div
                    self.deployment_request.is_deployment_split = True

                    avg_l = 0
                    avg_c = 0
                    avg_cpu = 0

                    for d in range(len(div)):
                        # Update allocated amounts
                        self.allocated_cpu[d] += div[d] * self.deployment_request.cpu_request
                        self.allocated_memory[d] += div[d] * self.deployment_request.memory_request
                        avg_cpu += 100 * (self.allocated_cpu[d] / self.cpu_capacity[d])
                        # Update free resources
                        self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                        self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                        # Latency updates
                        avg_l += self.latency[d] * div[d]
                        self.increase_latency(d, 1.05)  # Increase latency by 5% for the deployed clusters

                        # Cost updates
                        type_id = self.cluster_type[d]
                        avg_c += DEFAULT_CLUSTER_TYPES[type_id]['cost'] * div[d]

                        # Load updates
                        self.avg_load_served[d] += div[d]
                    
                    avg_l = avg_l / self.deployment_request.num_replicas
                    avg_c = avg_c / self.deployment_request.num_replicas
                    avg_cpu = avg_cpu / self.deployment_request.num_replicas

                    self.avg_latency.append(avg_l)
                    self.avg_cost.append(avg_c)
                    self.avg_cpu_usage_percentage_cluster_selected.append(avg_cpu)

                    # Save expected latency and cost for the deployment request
                    self.deployment_request.expected_latency = avg_l
                    self.deployment_request.expected_cost = avg_c

                    self.enqueue_request(self.deployment_request)


    def get_reward(self):
        # Power consumption reward, for now is just inverse of the power consumption
        power_consumption_reward = 1.0 / (self.total_power_consumption + 1e-6)  # Avoid division by zero

        if self.penalty:
            # Penalization of 5% for each value
            last_latency = self.avg_latency[-1] if self.avg_latency else 500.0  # valore di default
            last_cost = self.avg_cost[-1] if self.avg_cost else 100.0
            last_gini = calculate_gini_coefficient(self.avg_load_served)
            penalized_reward = np.array([
                last_latency * 1.05,
                last_cost * 1.05,
                last_gini * 0.95,
                power_consumption_reward * 1.05
            ], dtype=np.float32)
            return penalized_reward

        lat = 0
        cost = 0
        # Compute latency reward
        if not self.deployment_request.is_deployment_split:
            c = self.deployment_request.deployed_cluster
            type_id = self.cluster_type[c]
            cost = DEFAULT_CLUSTER_TYPES[type_id]['cost']
            lat = self.latency[self.deployment_request.deployed_cluster]
        else:
            lat = self.deployment_request.expected_latency
            cost = self.deployment_request.expected_cost
        
        latency_reward = lat # Minimize latency
        latency_reward = 1 / (latency_reward + 1e-6)  # Avoid division by zero
        cost_reward = cost # Minimize cost
        cost_reward = 1 / (cost + 1e-6)  # Avoid division by zero

        # Compute Gini coefficient reward
        gini = calculate_gini_coefficient(self.avg_load_served)
        gini_reward = gini
        gini_reward = 1 - gini  # Minimize Gini coefficient, so we want it to be as low as possible

        #Return reward as a vector as required by the multi-objective environment
        return np.array([latency_reward, cost_reward, gini_reward, power_consumption_reward], dtype=np.float32)


    def render(self, mode='human'):
        # Render the environment to the screen or return a visual representation
        pass

    def check_if_cluster_is_full_after_full_deployment(self, action):
        total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
        total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request
        
        if(self.allocated_cpu[action] + total_cpu > 0.95 * self.cpu_capacity[action] or
           self.allocated_memory[action] + total_memory > 0.95 * self.memory_capacity[action]):
            return True
        
        return False
    
    # Double-check if the selected clusters are full (spread strategy)
    def check_if_clusters_are_full_after_split_deployment(self, div):
        for d in range(len(div)):
            total_cpu = self.deployment_request.cpu_request * div[d]
            total_memory = self.deployment_request.memory_request * div[d]

            if (self.allocated_cpu[d] + total_cpu > 0.95 * self.cpu_capacity[d]
                    or self.allocated_memory[d] + total_memory > 0.95 * self.memory_capacity[d]):
                #logging.info('[Check]: Cluster {} is full...'.format(d))
                return True

        return False
    
    def deployment_generator(self):
        deployment_list = get_c2e_deployment_list()
        n = self.np_random.integers(low=0, high=len(deployment_list))
        d = deployment_list[n - 1]

        if self.min_replicas == self.max_replicas:
            d.num_replicas = self.min_replicas
        else:
            d.num_replicas = self.np_random.integers(low=self.min_replicas, high=self.max_replicas)
        return d
    
    def enqueue_request(self, request: DeploymentRequest) -> None:
        heapq.heappush(self.running_requests, (request.departure_time, request))

    # Remove deployment request
    def dequeue_request(self):
        _, deployment_request = heapq.heappop(self.running_requests)
        
        if deployment_request.is_deployment_split:
            # logging.info("[Dequeue] Deployment is split...")
            for d in range(self.num_clusters):
                total_cpu = self.deployment_request.cpu_request * self.deployment_request.split_clusters[d]
                total_memory = self.deployment_request.memory_request * self.deployment_request.split_clusters[d]

                # Update allocate amounts
                self.allocated_cpu[d] -= total_cpu
                self.allocated_memory[d] -= total_memory

                # Update free resources
                self.free_cpu[d] = self.cpu_capacity[d] - self.allocated_cpu[d]
                self.free_memory[d] = self.memory_capacity[d] - self.allocated_memory[d]

                # Decrease Latency if replicas were there
                if total_cpu != 0:
                    self.decrease_latency(d, 1.10)  # only 10% if split
        else:
            # logging.info("[Dequeue] Deployment is not split...")
            n = deployment_request.deployed_cluster
            total_cpu = self.deployment_request.num_replicas * self.deployment_request.cpu_request
            total_memory = self.deployment_request.num_replicas * self.deployment_request.memory_request

            # Update allocate amounts
            self.allocated_cpu[n] -= total_cpu
            self.allocated_memory[n] -= total_memory

            # Update free resources
            self.free_cpu[n] = self.cpu_capacity[n] - self.allocated_cpu[n]
            self.free_memory[n] = self.memory_capacity[n] - self.allocated_memory[n]

            # Decrease Latency
            self.decrease_latency(n, 1.15)  # 15% max reduction

    
    # Select (random) the next deployment request
    def next_request(self) -> None:
        arrival_time = self.current_time + self.np_random.exponential(scale=1 / self.arrival_rate_r)
        departure_time = arrival_time + self.np_random.exponential(scale=self.call_duration_r)
        self.dt = departure_time - arrival_time
        self.current_time = arrival_time

        while True:
            if self.running_requests:
                next_departure_time, _ = self.running_requests[0]
                if next_departure_time < arrival_time:
                    self.dequeue_request()
                    continue
            break

        self.deployment_request = self.deployment_generator()
        #logging.info('[Next Request]: Name: {} | Replicas: {}'.format(self.deployment_request.name, self.deployment_request.num_replicas))

    def increase_latency(self, n, factor):
        avg_value = self.latency[n]
        for n2 in range(self.num_clusters):
            if n == n2: # for the same node assume 0
                self.latency_matrix[n][n2] = 0
            else:
                prev = self.latency_matrix[n][n2]
                new_latency = max(min(prev * factor, MAX_DELAY), MIN_DELAY)

                self.latency_matrix[n][n2] = new_latency

                if self.latency_matrix[n][n2] == 0:
                    self.latency_matrix[n][n2] = MIN_DELAY  # Ensure no zero latency

                self.latency_matrix[n2][n] = self.latency_matrix[n][n2]  # Symmetric latency matrix

        # Update the average latency of all nodes
        for c in range(self.num_clusters):
            self.latency[c] = mean(self.latency_matrix[c])

     # Decrease Latency in the episode
    def decrease_latency(self, n, factor):
        avg_value = self.latency[n]
        for n2 in range(self.num_clusters):
            if n == n2:  # for the same node assume 0
                self.latency_matrix[n][n2] = 0
            else:
                prev = self.latency_matrix[n][n2]
                new_latency = max(min(prev / factor, MAX_DELAY), MIN_DELAY)

                self.latency_matrix[n][n2] = new_latency

                if self.latency_matrix[n][n2] == 0:
                    self.latency_matrix[n][n2] = 1.0

                self.latency_matrix[n2][n] = self.latency_matrix[n][n2]

                # logging.info("[Decrease Latency] previous latency: {} "
                #             "| updated Latency: {}".format(prev, self.latency_matrix[n][n2]))

        # Update Latency of all nodes
        for c in range(self.num_clusters):
            self.latency[c] = mean(self.latency_matrix[c])


    # Current Strategy: First Fit Decreasing (FFD)
    def first_fit_decreasing_heuristic(self, num_replicas, cpu_req, mem_req, num_clusters, free_cpu, free_mem):
        distribution = [0] * num_clusters

        # Min and max replicas per cluster
        min_replicas = 1
        max_replicas = num_replicas 

        # Distribute replicas across clusters
        for n in range(num_clusters):
            self.split_number_replicas[n] = min(free_cpu[n] // cpu_req, 
                                                free_mem[n] // mem_req)
            
        min_factor = int(math.ceil(min(self.split_number_replicas)))
        if min_factor > max_replicas:
            min_factor = max_replicas - 1 # To really distribute at the end

        # Sort the clusters by their remaining capacity (CPU) in decreasing order
        sorted_clusters_cpu = {}
        for n in range(num_clusters):
            sorted_clusters_cpu[str(n)] = free_cpu[n]

        sorted_clusters_cpu = sort_dict_by_value(sorted_clusters_cpu, reverse=True)

        for key, value in sorted_clusters_cpu.items():
            n = int(key)
            if num_replicas == 0:
                break
            if num_replicas > 0 and min_factor < num_replicas and (cpu_req * min_factor < value) and (mem_req * min_factor < free_mem[n]):
                distribution[n] += min_factor
                num_replicas -= min_factor   
            elif num_replicas > 0 and (cpu_req < value) and (mem_req < free_mem[n]):
                distribution[n] += min_replicas
                num_replicas -= min_replicas

        # Still distribute remaining replicas if needed 
        if num_replicas == 0:
            #logging.info('[first_fit_decreasing_heuristic] Replicas division: {}'.format(distribution))
            return distribution
        else:
            #logging.info('[first_fit_decreasing_heuristic] Replicas still to distribute...')
            for n in range(num_clusters):
                if num_replicas == 0:
                    break

                if (cpu_req < free_cpu[n]) and (mem_req < free_mem[n]):
                    distribution[n] += min_replicas
                    num_replicas -= min_replicas

            #logging.info('[first_fit_decreasing_heuristic] Replicas division: {}'.format(distribution))
            return distribution
        
    def get_state(self):
        # Get observation state
        cluster = np.full(shape=(NUM_SPREADING_ACTIONS + 1, NUM_METRICS_CLUSTER + 1), fill_value=-1)

        observation = np.stack([self.allocated_cpu,
                                self.cpu_capacity,
                                self.allocated_memory,
                                self.memory_capacity,
                                self.latency],
                                axis=1)
        
        # Condition the elements in the set with the current node requests
        request_demands = np.tile(
            np.array([self.deployment_request.num_replicas,
                 self.deployment_request.cpu_request,
                 self.deployment_request.memory_request,
                 self.deployment_request.latency_threshold,
                 self.dt]
            ),
            (self.num_clusters + NUM_SPREADING_ACTIONS + 1, 1)
        )

        observation = np.concatenate([observation, cluster], axis=0)
        observation = np.concatenate([observation, request_demands], axis=1)

        return observation
        
    def read_power_consumption(self, filename='./gym-multi-k8s/envs/kepler_power_consumption.csv'): 
        """
        Read power consumption from a file. We are using a CSV file with the following format:
        load,vWall,eCluster
        10,25,20
        20,35,35
        30,50,50
        where load is the CPU load, and vWall and eCluster are two different nodes type.
        """
        data = pd.read_csv(filename)
        data = data.set_index('load')
        return data
    
    def interpolate_power_consumption(self, load, node_type='vwall'):
        """
        Interpolate power consumption based on the load.
        vWall and eCluster are two different values for power conumption
        load is a number between the one listed in the table, e.g.:
        load,vWall,eCluster
        10,25,20
        20,35,35
        30,50,50
        Given a load of 15, it will return the interpolated values for vWall and eCluster.
        If the load is below the minimum or above the maximum, it will return the values for the closest load.
        :param load: CPU load
        :return: interpolated power consumption
        """
        #print(f"Interpolating power consumption for load: {load}")
        if load < 10:
            return self.power_consumption_data.loc[self.power_consumption_data.index.min(), node_type]
        elif load > 100:
            return self.power_consumption_data.loc[self.power_consumption_data.index.max(), node_type]
        else:
            # Interpolate the power consumption based on the load
            return np.interp(load, self.power_consumption_data.index, 
                             self.power_consumption_data[node_type].values)
            
    def calculate_power_consumption(self):
        """ 
        Calculate the power consumption based on the current load of the clusters.
        TODO: Let's add a cluster type parameter here, so we can discriminate 
        between two tier of resources, e.g. vWall and eCluster.
        """
        total_power_consumption = 0.0
        for c in range(self.num_clusters):
            # Get the CPU load for the current cluster
            cpu_load = (self.allocated_cpu[c] / self.cpu_capacity[c]) * 100.0  # Convert to percentage
            # convert CPU to a scalar value in the scale from 10 to 100
            device_profile = DEFAULT_CLUSTER_TYPES[self.cluster_type[c]]['device']
            cluster_consumption = self.interpolate_power_consumption(cpu_load, device_profile)
            #print(f"Cluster {c}, load: {cpu_load}, power consumption: {cluster_consumption} for device {device_profile}")
            self.cluster_power_consumption[c] = cluster_consumption
            total_power_consumption += cluster_consumption  # Assuming cost is power consumption per CPU unit
        self.total_power_consumption = total_power_consumption

    def calculate_incremental_power_consumption(self, cpu, cluster_index):
        """
        Calculate the incremental power consumption based on the current load of the clusters.
        :param cpu: CPU load
        :param cluster_index: Cluster index 
        :return: incremental power consumption
        """
        # Get the device 
        old_power_consumption = self.total_power_consumption
        return self.calculate_power_consumption() - old_power_consumption
