import sys
sys.path.append('..')
from nefesi.evaluation_scripts.evaluate_with_config import main
import datetime
if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    delta_time = end_time-start_time
    print("The total running time is: %.3f hours" % (delta_time.total_seconds()/3600))
