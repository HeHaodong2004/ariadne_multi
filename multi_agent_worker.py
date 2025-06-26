import matplotlib.pyplot as plt
from copy import deepcopy

from env import Env
from agent import Agent
from utils import *
from node_manager import NodeManager

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class MultiAgentWorker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.n_agent = N_AGENTS
        self.node_manager = NodeManager(plot=self.save_image)

        self.robot_list = [Agent(i, policy_net, self.node_manager, self.device, self.save_image) for i in
                           range(N_AGENTS)]

        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    def run_episode(self):
        done = False
        for robot in self.robot_list:
            robot.update_graph(self.env.belief_info, self.env.robot_locations[robot.id])
        for robot in self.robot_list:    
            robot.update_planning_state(self.env.robot_locations)

        for i in range(MAX_EPISODE_STEP):
            selected_locations = []
            dist_list = []
            next_node_index_list = []
            for robot in self.robot_list:
                observation = robot.get_observation()
                robot.save_observation(observation)

                next_location, next_node_index, action_index = robot.select_next_waypoint(observation)
                robot.save_action(action_index)

                node = robot.node_manager.nodes_dict.find((robot.location[0], robot.location[1]))
                check = np.array(node.data.neighbor_list)
                assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location,
                                                                                                         robot.location,
                                                                                                         node.data.neighbor_list)
                assert next_location[0] != robot.location[0] or next_location[1] != robot.location[1]

                selected_locations.append(next_location)
                dist_list.append(np.linalg.norm(next_location - robot.location))
                next_node_index_list.append(next_node_index)

            selected_locations = np.array(selected_locations).reshape(-1, 2)
            arriving_sequence = np.argsort(np.array(dist_list))
            selected_locations_in_arriving_sequence = np.array(selected_locations)[arriving_sequence]

            # solve collision
            for j, selected_location in enumerate(selected_locations_in_arriving_sequence):
                solved_locations = selected_locations_in_arriving_sequence[:j]
                while selected_location[0] + selected_location[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                    id = arriving_sequence[j]
                    nearby_nodes = self.robot_list[id].node_manager.nodes_dict.nearest_neighbors(
                        selected_location.tolist(), 25)
                    for node in nearby_nodes:
                        coords = node.data.coords
                        if coords[0] + coords[1] * 1j in solved_locations[:, 0] + solved_locations[:, 1] * 1j:
                            continue
                        selected_location = coords
                        break

                    selected_locations_in_arriving_sequence[j] = selected_location
                    selected_locations[id] = selected_location

            reward_list = []
            for robot, next_location, next_node_index in zip(self.robot_list, selected_locations, next_node_index_list):
                self.env.step(next_location, robot.id)
                individual_reward = robot.utility[next_node_index] / (SENSOR_RANGE * 3.14 // FRONTIER_CELL_SIZE)
                reward_list.append(individual_reward)

                robot.update_graph(self.env.belief_info, deepcopy(self.env.robot_locations[robot.id]))

            if self.robot_list[0].utility.sum() == 0:
                done = True

            team_reward = self.env.calculate_reward() - 0.5
            if done:
                team_reward += 10

            for robot, reward in zip(self.robot_list, reward_list):
                robot.save_reward(reward + team_reward)
                robot.update_planning_state(self.env.robot_locations)
                robot.save_done(done)

            if self.save_image:
                self.plot_env(i)

            if done:
                break

        # save metrics
        self.perf_metrics['travel_dist'] = max([robot.travel_dist for robot in self.robot_list])
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = done

        # save episode buffer
        for robot in self.robot_list:
            observation = robot.get_observation()
            robot.save_next_observations(observation)
            for i in range(len(self.episode_buffer)):
                self.episode_buffer[i] += robot.episode_buffer[i]

        # save gif
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)

    def plot_env(self, step):
        plt.switch_backend('agg')
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 2)
        plt.imshow(self.env.robot_belief, cmap='gray')
        plt.axis('off')
        color_list = ['r', 'b', 'g', 'y']

        for robot in self.robot_list:
            c = color_list[robot.id]
            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)
            plt.plot((np.array(robot.trajectory_x) - robot.map_info.map_origin_x) / robot.cell_size,
                     (np.array(robot.trajectory_y) - robot.map_info.map_origin_y) / robot.cell_size, c,
                     linewidth=2, zorder=1)

        plt.subplot(1, 2, 1)
        plt.imshow(self.env.robot_belief, cmap='gray')
        for robot in self.robot_list:
            c = color_list[robot.id]
            if robot.id == 0:
                nodes = get_cell_position_from_coords(robot.node_coords, robot.map_info)
                plt.imshow(robot.map_info.map, cmap='gray')
                plt.axis('off')
                plt.scatter(nodes[:, 0], nodes[:, 1], c=robot.utility, zorder=2)

            robot_cell = get_cell_position_from_coords(robot.location, robot.map_info)
            plt.plot(robot_cell[0], robot_cell[1], c+'o', markersize=16, zorder=5)

            if robot.id == 0:
                for coords in robot.node_coords:
                    node = self.node_manager.nodes_dict.find(coords.tolist()).data
                    for neighbor_coords in node.neighbor_list[1:]:
                        end = (np.array(neighbor_coords) - coords) / 2 + coords
                        plt.plot((np.array([coords[0], end[0]]) - robot.map_info.map_origin_x) / robot.map_info.cell_size,
                                       (np.array([coords[1], end[1]]) - robot.map_info.map_origin_y) / robot.map_info.cell_size, 'tan', zorder=1)

        if len(self.env.global_frontiers) > 0:
            frontiers = get_cell_position_from_coords(np.array(list(self.env.global_frontiers)), self.env.belief_info).reshape(-1, 2)
            plt.scatter(frontiers[:, 0], frontiers[:, 1], c='r', s=2)

        plt.axis('off')
        plt.suptitle('Explored ratio: {:.4g}  Travel distance: {:.4g}'.format(self.env.explored_rate,
                                                                              max([robot.travel_dist for robot in
                                                                                   self.robot_list])))
        plt.tight_layout()
        # plt.show()
        plt.savefig('{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step), dpi=150)
        frame = '{}/{}_{}_samples.png'.format(gifs_path, self.global_step, step)
        self.env.frame_files.append(frame)
        plt.close()