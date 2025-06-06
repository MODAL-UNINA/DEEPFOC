      subroutine range (prompt, value, vmin, vmax)

      character         prompt*(*)                      
c							! (input) prompt string
      real              value                           
c							! (input/output) value
      real              vmax                            
c							! (input) maximum value
      real              vmin                            
c							! (input) minimum value

      real              askr                            
c							! function

10    value = askr (prompt, value)
      if (value .lt. vmin) then
        write (*, 100) vmin
100     format (/' value must be greater than ', g11.5, '; try again')
        goto 10
      else if (vmax .gt. 0. .and. value .gt. vmax) then
        write (*, 110) vmax
110     format (/' value must be less than ', g11.5, '; try again')
        goto 10
      end if
      return
      end
