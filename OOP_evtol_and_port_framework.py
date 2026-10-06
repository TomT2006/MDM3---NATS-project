import simpy
import random
import statistics

wait_times =[]

class airport:
    
    def __init__(self, env, num_TLOFs, num_parking_bays):

        self.env = env
        self.TLOFs = simpy.Resource(env, capacity = num_TLOFs) # can be changed to PriorityResource in future for emergency implementation
        #dont know yet how many aircraft can fit in taxi queue
        #however can just use this framework for aircraft requesting TLOF and entering queue without actually taxxiing
        self.parking_bays = simpy.Resource(env, num_parking_bays)
        self.energy = simpy.Container(env, init=1000) # kWh

class eVTOL:

    def __init__(self, env, type, state, battery_level, range, queue_position):
        #
        self.env = env
        self.type = None # Tilt rotor, Lift + Cruise, Multi-copter or Vectored Thrust (not useful for an initial model but may be implemented in future)
        self.state = None # Idle, Taxiing, Takeoff, Cruising, Landing (assuming idling time is charging time)
        self.battery_level = simpy.Container(env, init=battery_level, capacity=battery_level) # kWh
        self.range = range # km
        self.queue_position = None # position in the queue for takeoff/landing


    def arrival_process(self, airport):
        #1. requests TLOF zone resource from airport for landing
        with airport.TLOFs.request() as tlof_req:
            yield tlof_req
            print(f"eVTOL {self.queue_position} is landing at {self.env.now}")
            yield self.env.timeout(random.randint(1, 5)) # Simulate landing time
            print(f"eVTOL {self.queue_position} has landed at {self.env.now}")

            # 2. while on TLOF pad, request parking bay 
            self.parking_bay = airport.parking_bays.request()
            yield self.parking_bay
            print(f"eVTOL {self.queue_position} obtained parking bay, starting taxi to parking bay at {self.env.now}")
            yield self.env.timeout(2)

        print(f"eVTOL {self.queue_position} clear of TLOF and safely parked at {self.env.now}")


    def takeoff_process(self, airport):
        #requests TLOF zone resource from airport for takeof
        with airport.TLOFs.request() as tlof_req:
            yield tlof_req
            print(f"eVTOL {self.queue_position} taxiing out to TLOF at {self.env.now}")
            yield self.env.timeout(1)# taxi time

            airport.parking_bays.release(self.parking_bay) # release the parking bay
            self.parking_bay = None

            print(f"eVTOL {self.queue_position} is taking off at {self.env.now}")
            yield self.env.timeout(2) # takeoff time
            print(f"eVTOL {self.queue_position} has taken off at {self.env.now}")



