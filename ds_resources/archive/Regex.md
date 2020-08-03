##### start of the line: ^
`^`i think

##### end of the line: $
morning`$`

##### character class: []
Any character inside the brackets:
`[Bb][Uu][Ss][Hh]`

Range:
`[0-9][a-z][A-Z]`

##### excluding character class: [^ ]
`[^?.^]`$ _exclude ? and . and ^ at the end of lines_

##### any character: .
9`.`11 _the dot is a wildcard: any character will do_

##### x or y: |
flood`|`fire`|`earthquake _any number of alternatives_

##### optional expression: ()?
`colou?r`: color or colour  
[Gg]eorge`( [Ww]\.)?` [Bb]ush

##### any number, including none: \*
`(.*)` _any character, including none_  
`\( (.*) \)` _any character between parentheses, including none_  
`\( ([^\)]+) \)` _any sequence between parentheses that do not include parentheses themselves_

##### any number, including one: +

##### greedy vs non greedy: \+? and \*?
Note that `+` and `*` are greedy, so they will match the longest string matching the regex.
`(.*?)` for non greedy terms, ie. the first string that matches the regex.

##### number of occurences
[Bb]ush`( +[^ ]+){1,5}` debate _Bush followed by at least one word and at most five, then debate_

##### repeting occurences
` +([a-zA-Z]+) +\1 +` _space, then twice the value of the parenthesis, then a space_
 

##### Combining characters
^[Ii] think  
^[0-9][a-zA-Z]  
^[Gg]ood|[Bb]ad _good at the beginning or bad anywhere_  
^([Gg]ood|[Bb]ad) _good or bad at the beginning_

##### Isolate words
`[^A-Za-z][Tt]he[^A-Za-z]`: isolate The & the

##### Lookahead (?!...)

`regex (?!_)`: regex not immediately followed by `_`.