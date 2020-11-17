uint8 state
uint8 INIT = 0		# Waiting for event to be selected
uint8 READY = 1		# Ready for GO signal
uint8 MAPPING = 2	# Slow lap for mapping (track drive only)
uint8 RUNNING = 3	# Full speed driving
uint8 FINISHED = 4	# Finished necessary laps
uint8 STOPPED = 5	# Come to a complete stop
uint8 EMERGENCY = 9	# Something bad happened; enable emergency brakes
