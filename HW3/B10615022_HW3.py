import sys

line_count = 0
num_of_q = 0
q_result = []
q_relate = []
for line in sys.stdin:
    if line_count == 0:
        num_of_q = int(line)
        line_count += 1

    else:
        if line_count % 2 == 0:
            q_relate.append(line.split())
        else:
            q_result.append(line.split())
        line_count += 1


_map = 0.0
for i in range(num_of_q):
    ap = 0.0
    results = q_result[i]
    relates = q_relate[i]
    idx = 0
    count = 0
    for result in results:
        if result in relates:
            count += 1
            ap += count / (idx + 1)
        idx += 1
        
            
    ap /= len(relates)
    _map += ap


print(round(_map/num_of_q,4))

            

