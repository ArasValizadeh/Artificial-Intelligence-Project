#!/bin/bash
#run with bash test.sh

# Initialize counters for the wins
agent1_wins=0
agent2_wins=0

# Loop 10 times
for i in {1..10}
do
    echo "🚀 Running game $i"
    result=$(python3 main.py)

    # Check the output and increment the counters
    if [[ "$result" == *"CodeCalak"* ]]; then
        echo "😀 result: $result CodeCalak wins"
        ((agent1_wins++))
    elif [[ "$result" == *"random"* ]]; then
        echo "😀 result: $result random wins"
        ((agent2_wins++))
    fi
done

# Print the results
echo "CodeCalak wins: $agent1_wins"
echo "random wins: $agent2_wins"