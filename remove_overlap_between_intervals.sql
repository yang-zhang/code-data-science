-- Intervals A:
-- 1 2 3 4 5 6 7 8 9
-- 1 - - 4 
--         5 - - - 9

-- Intervals B:
-- 1 2 3 4 5 6 7 8 9
--   2 - - - 6  
--             7 8 

-- Remove Intervals B from Intervals A:
-- 1 2 3 4 5 6 7 8 9
-- 1 2               
--               8 9

--
drop table if exists merge_overlap_o;
create table merge_overlap_o (t_start integer, t_end integer);
insert into merge_overlap_o values
(1, 4),
(5, 9);

--
drop table if exists merge_overlap_m;
create table merge_overlap_m (t_start integer, t_end integer);
insert into merge_overlap_m values
(2, 6),
(7, 8);

--
select t_start,
min(t_end) as t_end
from 
(
  select 
  s1.t_start as t_start,
  least(s1.t_end, t1.t_start) as t_end 
  from merge_overlap_o s1 
  join merge_overlap_m t1 
  on s1.t_start < least(s1.t_end, t1.t_start)
  where not exists (select * from merge_overlap_m m where s1.t_start >= m.t_start and s1.t_start < m.t_end)
)
group by t_start

union

select max(t_start) as t_start,
t_end
from
(
  select 
  t1.t_end as t_start,
  s1.t_end as t_end 
  from merge_overlap_o s1 
  join merge_overlap_m t1 
  on t1.t_end < s1.t_end
)
group by t_end;

--
select t_start,
min(t_end) as t_end
from 
(
  select 
  s1.t_start as t_start,
  least(s1.t_end, t1.t_start) as t_end 
  from merge_overlap_o s1 
  join merge_overlap_m t1 
  on s1.t_start < least(s1.t_end, t1.t_start)
  where not exists (select * from merge_overlap_m m where s1.t_start >= m.t_start and s1.t_start < m.t_end)
)
group by t_start

union

select max(t_start) as t_start,
t_end
from
(
  select 
  t1.t_end as t_start,
  s1.t_end as t_end 
  from merge_overlap_o s1 
  join merge_overlap_m t1 
  on t1.t_end < s1.t_end
)
group by t_end;












