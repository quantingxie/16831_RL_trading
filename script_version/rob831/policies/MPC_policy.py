import numpy as np

from .base_policy import BasePolicy

# MPC is a gradient-free method, so it doesn't have update and forward functions. 
class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]
        # action space
        # The actions space and observation spaces are all defined in environment settings
        # Spaces have data structures like Box(2,) or Discrete(2,), 
        # those are the basic data structures provided by gym to represent contineous and 
        # discrete spaces. 

        # Box(n,) corresponds to n-dimentiosnal contineous space. 
        self.ac_space = self.env.action_space
        
        self.ac_dim = ac_dim

        # What are the high and lows of action spaces? 
        # They represents the high bound and low bound of the value of our actions 
        self.low = self.ac_space.low
        self.high = self.ac_space.high


        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def get_random_actions(self, num_sequences, horizon):
        # TODO(Q1) uniformly sample actions and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]

        # print ("ssssssss", self.ac_dim)
        # acts = self.low + np.random.random(
        #     (num_sequences, horizon, self.ac_dim)) * (self.high - self.low)
        acts = np.random.normal(loc = (self.high + self.low)/2, scale = 1, size= (num_sequences,horizon, self.ac_dim))
        # print("random_action_sequence", acts)
        # print("N", self.N)
        # print("H", self.horizon)
        #acts = np.random.randint(2, size=(num_sequences, horizon, self.ac_dim))
        return acts

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' \
            or (self.sample_strategy == 'cem' and obs is None):
            random_action_sequences = self.get_random_actions(num_sequences, horizon)# TODO(Q1) sample random actions
            return random_action_sequences

        elif self.sample_strategy == 'cem':
            # TODO(Q5): Implement action selection using CEM.
            # Begin with randomly selected actions, then refine the sampling distribution
            # iteratively as described in Section 3.3, "Iterative Random-Shooting with Refinement" of
            # https://arxiv.org/pdf/1909.11652.pdf

            #initialize CEM distribution and initial actions
            acs = self.get_random_actions(num_sequences, horizon)
            
            elite_mean = np.mean(acs, axis=0)
            elite_std = np.std(acs, axis=0)

            for i in range(self.cem_iterations):
                acs = np.random.normal(loc = elite_mean,scale = elite_std, size = (num_sequences,horizon, self.ac_dim))

                # - Sample candidate sequences from a Gaussian with the current
                #   elite mean and variance
                #     (Hint: remember that for the first iteration, we instead sample
                #      uniformly at random just like we do for random-shooting)

                rewards = self.evaluate_candidate_sequences(acs, obs)
                #print ("rewards ----", np.shape(rewards))
                ind = np.argpartition(rewards, -self.cem_num_elites)[-self.cem_num_elites:]
                #print ("ind--------", np.shape(ind))
                elite_actions = acs[ind]
                #print ("elite_actions", np.shape(elite_actions))
                
                elite_mean_next = self.cem_alpha * np.mean(elite_actions, axis=0) + (1-self.cem_alpha) * elite_mean
                elite_std_next = self.cem_alpha * np.std(elite_actions, axis=0) + (1-self.cem_alpha) * elite_std

                #print ("elite_mean", np.shape(elite_mean))

                elite_mean = elite_mean_next
                elite_std = elite_std_next
                # - Get the top `self.cem_num_elites` elites
                #     (Hint: what existing function can we use to compute rewards for
                #      our candidate sequences in order to rank them?)
                # - Update the elite mean and variance
                
                

            # TODO(Q5): Set `cem_action` to the appropriate action sequence chosen by CEM.
            # The shape should be (horizon, self.ac_dim)
            cem_action = np.mean(elite_actions,axis=0)
            #print("cem_action", np.shape(cem_action))

            return cem_action[None]
        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")

    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): for each model in ensemble, compute the predicted sum of rewards
        # for each candidate action sequence.
        #
        # Then, return the mean predictions across all ensembles.
        # Hint: the return value should be an array of shape (N,)
        reward_list = []
        for model in self.dyn_models:
            rewards = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            reward_list.append(rewards)

        mean = np.mean(reward_list, axis= 0)
        return mean

    def get_action(self, obs):
        if self.data_statistics is None:


            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM: only a single action sequence to consider; return the first action
            # print ("best_action_sequence", candidate_action_sequences)

            return candidate_action_sequences[0][0][None]

        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)
            
            # pick the action sequence and return the 1st element of that sequence
            best_action_sequence = candidate_action_sequences[np.argmax(predicted_rewards)] # TODO (Q2)
            action_to_take = best_action_sequence[0] # TODO (Q2)
            # print ("best_action_sequence", best_action_sequence)
            return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs, action)` at each step.
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.


        N, H, D_actions = candidate_action_sequences.shape

        # Initilzie (N,) place holders for reward
        sum_of_rewards = np.zeros(self.N)  # TODO (Q2) 

        # np.tile repeat arrays, in this case, it repeat the obs (N, 1) times
        # Repeat current observation n times to evaluate each action sequences
        predicted_obs = np.tile(obs, (N, 1))
        # print("predicted_obs from Calculate_reward, MPC_policy", predicted_obs.shape)

        
        for h in range(H):
            # Calculate rewards between action sequences at the same time
            actions = candidate_action_sequences[:, h, :]
            # Use reward function in env to simulate the total rewards given generated actions and predicted observation by model
            #rewards, _ = self.env.get_reward(predicted_obs, actions)
            # Changed to _calculate_reward because stock env doesn't have the get reward function
            rewards, _ = self.env.get_reward(predicted_obs, actions)
            # print("predicted_obs from Calculate_reward, MPC_policy", predicted_obs.shape)
        
            sum_of_rewards += rewards
            predicted_obs = model.get_prediction(predicted_obs, actions, self.data_statistics)
            # print("predicted_obs from Calculate_reward, MPC_policy", predicted_obs.shape)

        return sum_of_rewards
