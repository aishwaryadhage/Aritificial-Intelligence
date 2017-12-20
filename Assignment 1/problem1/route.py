#!/usr/bin/env python
# The report for this problem is saved in report.md file which is in the same directory as this file.

import heapq
from collections import defaultdict
import re
import sys
from math import sin, cos, sqrt, atan2, radians

class ShortestPath:
    def __init__(self, start_city, end_city):
        self.d_cities = defaultdict (list)  # dictionary to store the cities

        self.start_city = start_city
        self.end_city = end_city
        self.cities = [ ]
        self.stack_city = [ ]
        self.stack_speed = [ ]
        self.distance = [ ]
        self.fringe = [ ]
        self.newpaths = [ ]
        # A* Search variables
        self.cities_length = [ ]
        self.d_astar_distance = defaultdict (list)

    # reads file from road-segments.txt
    def readfile(self):
        with open ("road-segments.txt") as fobj:
            for line in fobj:
                self.cities.append (re.sub (' ', ' ', line).split ())

    # reads file from city-gps.txt
    def read_heuristicfile(self):
        with open ("city-gps.txt") as fobj1:
            for line in fobj1:
                self.cities_length.append (re.sub (' ', ' ', line).split ())

    # Using Haversine formula to calculate the distance between two points using latitude and longitude
    def heuristic(self, start_city, end_city):
        self.read_heuristicfile ()
        distance_miles = 0
        for i in range (len (self.cities_length)):
            if start_city == self.cities_length[ i ][ 0 ]:
                radius = 6373.0  # approx radius of earth
                latitude_startcity, longitude_startcity, latitude_endcity, longitude_endcity = 0, 0, 0, 0
                # lattitude and longitude for start city
                for i in range (len (self.cities_length)):
                    if start_city == self.cities_length[ i ][ 0 ]:
                        latitude_startcity = radians (float (self.cities_length[ i ][ 1 ]))
                        longitude_startcity = radians (float (self.cities_length[ i ][ 2 ]))

                # lattitude and longitude for end city
                for j in range (len (self.cities_length)):
                    if end_city == self.cities_length[ j ][ 0 ]:
                        latitude_endcity = radians (float (self.cities_length[ j ][ 1 ]))
                        longitude_endcity = radians (float (self.cities_length[ j ][ 2 ]))

                dlon = longitude_endcity - longitude_startcity
                dlat = latitude_endcity - latitude_startcity

                angle = sin (dlat / 2) ** 2 + cos (latitude_startcity) * cos (latitude_endcity) * sin (dlon / 2) ** 2
                curve = 2 * atan2 (sqrt (angle), sqrt (1 - angle))

                distance = radius * curve
                distance_miles = distance * 0.621371    # convert km into miles
        return float ("{0:.2f}".format (distance_miles))

    # creates a dictionary to all the corresponding cities using the start city
    def graph_generator(self, start_city):
        self.readfile ()
        self.stack_city.append (start_city)
        while len (self.stack_city) > 0:
            appendCities = [ ]
            del appendCities[ : ]
            city = self.stack_city.pop (0)
            for i in range (len (self.cities)):
                if city == self.cities[ i ][ 0 ]:
                    self.stack_city.append (self.cities[ i ][ 1 ])
                    appendCities.append ([ self.cities[ i ][ 1 ], self.cities[ i ][ 2 ], self.cities[ i ][ 3 ] ])
                    for j in range (len (self.cities)):
                        if self.cities[ i ][ 1 ] == self.cities[ j ][ 1 ]:
                            self.d_cities[ self.cities[ i ][ 1 ] ]. \
                                append ([ self.cities[ j ][ 0 ], self.cities[ j ][ 2 ], self.cities[ j ][ 3 ] ])
            self.d_cities[ city ] += appendCities[ : ]
        return self.d_cities

    # Function used for both DFS and BFS search
    def search(self, graph, start, goal, algorithm):
        distance = 0
        time = 0
        visited = {start: None}
        queue = ([ start ])
        while queue:
            if algorithm == 'bfs':
                node = queue.pop (0)
            elif algorithm == 'dfs':
                node = queue.pop ()
            if node == goal:
                path = [ ]
                while node is not None:
                    path.append (node)
                    node = visited[ node ]

                path_modified = path[ ::-1 ]
                for i in range (0, len (path) - 1):
                    for j in range (len (self.cities)):
                        if (path_modified[ i ] == self.cities[ j ][ 0 ] and path_modified[ i + 1 ] == self.cities[ j ][
                            1 ]) \
                                or (path_modified[ i ] == self.cities[ j ][ 1 ] and path_modified[ i + 1 ] ==
                                    self.cities[ j ][ 0 ]):
                            distance += int (self.cities[ j ][ 2 ])
                            time += int (self.cities[ j ][ 3 ])
                speed = float ("{0:.2f}".format (float (distance) / float (time)))
                path.append (speed)
                path.append (distance)
                return path[ ::-1 ]

            for neighbour in graph[ node ]:
                if neighbour[ 0 ] not in visited:
                    visited[ neighbour[ 0 ] ] = node
                    queue.append (neighbour[ 0 ])
        return "NotFound"

    # This function is used for both A* and Uniform cost function
    def a_star_search(self, graph, start, goal, algorithm, cost_function):
        fringe = [ ]
        heapq.heappush (fringe, (0, start))
        visited = {}
        cost_calculate = {}
        visited[ start ] = None
        cost_calculate[ start ] = 0

        # Condition for astar algorithm with all the cost function in the inner if condition
        if algorithm == 'astar':
            while not len (fringe) == 0:
                current = heapq.heappop (fringe)[ 1 ]
                if current == goal:
                    break
                for next in graph[ current ]:
                    if cost_function == 'distance':
                        new_cost = cost_calculate[ current ] + int (next[ 1 ])
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost + self.heuristic (next[ 0 ], goal)
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current
                    elif cost_function == 'time':
                        new_cost = cost_calculate[ current ] + int (next[ 2 ])
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost + self.heuristic (next[ 0 ], goal)
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current
                    elif cost_function == 'segments':
                        new_cost = cost_calculate[ current ] + 1
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost + self.heuristic (next[ 0 ], goal)
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current

        # Condition for astar algorithm with all the cost functions in the inner if condition
        elif algorithm == 'uniform':
            while not len (fringe) == 0:
                current = heapq.heappop (fringe)[ 1 ]
                if current == goal:
                    break
                for next in graph[ current ]:
                    if cost_function == 'distance':
                        new_cost = cost_calculate[ current ] + int (next[ 1 ])
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current
                    elif cost_function == 'time':
                        new_cost = cost_calculate[ current ] + int (next[ 2 ])
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current
                    elif cost_function == 'segments':
                        new_cost = cost_calculate[ current ] + 1
                        if next[ 0 ] not in cost_calculate or new_cost < cost_calculate[ next[ 0 ] ]:
                            cost_calculate[ next[ 0 ] ] = new_cost
                            priority = new_cost
                            heapq.heappush (fringe, (priority, next[ 0 ]))
                            visited[ next[ 0 ] ] = current
        current = goal
        path = [ current ]
        distance = 0
        time = 0
        while current != start:
            current = visited[ current ]
            path.append (current)
        modified_path = path[ ::-1 ]
        for i in range (0, len (path) - 1):
            for j in range (len (self.cities)):
                if (modified_path[ i ] == self.cities[ j ][ 0 ] and modified_path[ i + 1 ] == self.cities[ j ][ 1 ]) \
                        or (modified_path[ i ] == self.cities[ j ][ 1 ] and modified_path[ i + 1 ] == self.cities[ j ][
                            0 ]):
                    distance += int (self.cities[ j ][ 2 ])
                    time += int (self.cities[ j ][ 3 ])
        speed = float ("{0:.2f}".format (float (distance) / float (time)))
        path.append (speed)
        path.append (distance)
        return path[ ::-1 ]

# Takes the value from the commandline
start_city = sys.argv[ 1 ]
end_city = sys.argv[ 2 ]
algorithm = sys.argv[ 3 ]
cost_function = sys.argv[ 4 ]
initial = ShortestPath (start_city, end_city)
result = [ ]
if algorithm == 'bfs' or algorithm == 'dfs':
    result = initial.search (initial.graph_generator (start_city), start_city, end_city, algorithm)
elif algorithm == 'astar' or algorithm == 'uniform':
    result = initial.a_star_search (initial.graph_generator (start_city), start_city, end_city, algorithm,
                                    cost_function)
else:
    print ("You entered wrong input")
print (' '.join (str (v) for v in result))
