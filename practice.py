def parse_log(log: str, header: str) -> str:
    for line in log.splitlines():
        if header in line:
            return "found: " + line
    
    return "data not found"

log = """
color = blue
number = 12
name = gurt
"""

search_for = input("search for?\n")
print(parse_log(log, search_for))