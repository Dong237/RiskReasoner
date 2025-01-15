"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import cloudpickle
import numpy as np
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs):
        self.num_envs = num_envs

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError


def shareworker(remote, parent_remote, env_fn_wrapper):
    """
    The worker function that runs in a subprocess (here the subprocess means the
    child process specifically, since parent_conn is closed).

    It listens for commands from the main process to interact with its environment.

    Args:
        remote (Connection): The connection to communicate with the main process.
        parent_remote (Connection): The parent connection (closed in the worker).
        env_fn_wrapper (CloudpickleWrapper): The wrapped environment creation function.
    """
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        # the following lines are only then triggered after the parent_conn.recv is called
        if cmd == 'step':
            ob, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_num_agents':
            remote.send((env.n_agents))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send((fr))
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    """
    A vectorized environment that runs multiple environments in parallel subprocesses.

    This class manages the creation, communication, and termination of subprocesses 
    running individual environment instances.
    
    *Extra
    Main Process:
        for each remote, action in remotes, actions:
            remote.send(('step', action))
        wait for all remotes to receive and process steps
        for each remote in remotes:
            ob, rew, done, info = remote.recv()
        return aggregated results

    Subprocesses (one per env):
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, rew, done, info = env.step(data)
                remote.send((ob, rew, done, info))
            ...
    """
    def __init__(self, env_fns, spaces=None):
        """
        Initializes the ShareSubprocVecEnv.

        Args:
            env_fns (list): A list of callables, each returning a new environment instance.
            spaces (optional): The observation and action spaces (not used here).
        """
        self.waiting = False
        self.closed = False
        self.nenvs = len(env_fns)
        # parent_conn, chile_conn = Pipe()
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.nenvs)])
        # In the shareworker function, self.remotes will be closed
        # NOTE Each entry in self.ps is a child process (Process(...)).
        self.ps = [Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
            
        # Here self.work_remotes will be closed
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_num_agents', None))
        self.n_agents = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns))

    def step_async(self, actions):
        """
        Sends actions to all subprocess environments asynchronously.

        Args:
            actions (iterable): A batch of actions to apply to each environment.
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        """
        Waits for all subprocess environments to complete their steps.

        Returns:
            tuple: A tuple containing observations, rewards, dones, and infos from all environments.
        """
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, dones, infos = zip(*results)
        return np.stack(obs), np.stack(dones), infos

    def reset(self):
        """
        Resets all subprocess environments and retrieves initial observations.

        Returns:
            np.ndarray: Array of initial observations from all environments.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs = np.array(results)
        return np.stack(obs)

    def reset_task(self):
        """
        Resets the task for all subprocess environments.

        Returns:
            np.ndarray: Array of initial observations after task reset.
        """
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        """
        Closes all subprocess environments and terminates subprocesses.

        Ensures that all resources are cleaned up properly.
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class ShareDummyVecEnv(ShareVecEnv):
    """
    A vectorized environment that runs multiple environments in the same process.

    This class is useful for debugging or environments that are lightweight and 
    do not require the overhead of multiprocessing.
    """
    def __init__(self, env_fns):
        """
        Initializes the ShareDummyVecEnv.

        Args:
            env_fns (list): A list of callables, each returning a new environment instance.
        """
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.n_agents = env.n_agents
        ShareVecEnv.__init__(self, len(env_fns))
        self.actions = None

    def step_async(self, actions):
        """
        Stores the actions to be applied in the next step.

        Args:
            actions (iterable): A batch of actions to apply to each environment.
        """
        self.actions = actions

    def step_wait(self):
        """
        Applies the stored actions to all environments and retrieves results.

        Returns:
            tuple: A tuple containing observations, rewards, dones, and infos from all environments.
        """
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()
        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        """
        Resets all environments and retrieves initial observations.

        Returns:
            np.ndarray: Array of initial observations from all environments.
        """
        results = [env.reset() for env in self.envs]
        obs = np.array(results)
        return obs

    def close(self):
        """
        Closes all environments managed by this vectorized environment.
        """
        for env in self.envs:
            env.close()

    def save_replay(self):
        """
        Saves replay logs for all environments.

        This is useful for environments that support replay functionality.
        """
        for env in self.envs:
            env.save_replay()


