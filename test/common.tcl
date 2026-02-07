# Common utilities for UCI expect tests

# Default timeout for expect
set timeout 10

# Path to UCI binary (must be set via UCI_BINARY env var)
if {![info exists ::env(UCI_BINARY)] || $::env(UCI_BINARY) eq ""} {
    puts stderr "ERROR: UCI_BINARY environment variable not set."
    puts stderr "Run tests via: ./run_tests.sh <path-to-uci-binary>"
    exit 1
}
set UCI_BINARY $::env(UCI_BINARY)

# Test result tracking
set test_passed 0
set test_failed 0
set test_name ""

proc test_start {name} {
    global test_name
    set test_name $name
}

proc test_pass {{msg ""}} {
    global test_passed
    incr test_passed
}

proc test_fail {msg} {
    global test_failed test_name
    incr test_failed
    puts "FAIL: $test_name - $msg"
}

proc test_summary {} {
    global test_passed test_failed
    if {$test_failed > 0} {
        puts "$test_passed passed, $test_failed failed"
        return 1
    }
    return 0
}

# Start UCI process
proc start_uci {} {
    global UCI_BINARY spawn_id
    spawn $UCI_BINARY
    # Delay to ensure process is ready for input
    sleep 0.2
    return $spawn_id
}

# Send command and wait for specific response
proc uci_send {cmd} {
    send "$cmd\r"
}

# Wait for a pattern with timeout
proc uci_expect {pattern {timeout_val 10}} {
    set timeout $timeout_val
    expect {
        -re $pattern {
            return 1
        }
        timeout {
            return 0
        }
        eof {
            return -1
        }
    }
}

# Send command and expect response pattern
proc uci_cmd_expect {cmd pattern {timeout_val 10}} {
    uci_send $cmd
    return [uci_expect $pattern $timeout_val]
}

# Clean shutdown
proc uci_quit {} {
    uci_send "quit"
    expect {
        eof { return 1 }
        timeout { return 0 }
    }
}
