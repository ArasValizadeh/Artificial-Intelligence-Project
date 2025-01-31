#!/bin/bash
#run with bash test.sh

# Initialize counters for the wins
agent1_wins=0
agent2_wins=0
agent_name1="CodeKalakAgent2"
agent_name2="random_agent"

for i in {1..10}
do
    echo "🚀 Running game $i"
    result=$(python3 main.py --player1 "$agent_name1" --player2 "$agent_name2")

    # Check the output and increment the counters
    if [[ $result == *"$agent_name1"* ]]; then
        echo "😀 result: $result - $agent_name1 wins"
        ((agent1_wins++))
    elif [[ $result == *"$agent_name2"* ]]; then
        echo "😀 result: $result - $agent_name2 wins"
        ((agent2_wins++))
    fi
done

# Print the results
echo "Final Score:"
echo "$agent_name1 wins: $agent1_wins"
echo "$agent_name2 wins: $agent2_wins"
