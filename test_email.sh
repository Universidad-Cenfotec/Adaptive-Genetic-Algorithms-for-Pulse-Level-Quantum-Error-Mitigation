#!/bin/bash

# Email recipient (replace with your email)
EMAIL="wil-19-60@live.com"

# Function to send an email
send_email() {
    local SUBJECT="$1"
    local MESSAGE="$2"
    echo -e "$MESSAGE" | mail -s "$SUBJECT" "$EMAIL"
}

# Simulated command (this will fail)
echo "🔧 Running test command..."
ls /non_existent_directory > /dev/null 2>&1
EXIT_CODE=$?

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "❌ Test command failed!"
    send_email "⚠️ Test Failure" "Test command failed!\nCheck your system logs."
else
    echo "✅ Test command succeeded!"
    send_email "✅ Test Success" "Test command completed successfully!"
fi

echo "🎉 All tests completed!"
send_email "🚀 Test Email System Completed" "All test emails have been sent. Check your inbox!"
