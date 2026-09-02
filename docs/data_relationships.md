# Data Relationships Document

This document describes the schema relations, primary keys, foreign keys, and analytical relationships among the processed files in `data/processed/`.

## 1. Processed Datasets Overview

The following datasets exist in `data/processed/` at this stage:
- **`employees.csv`**: Core employee table (Employee Master).
- **`engagement_data.csv`**: Survey engagement scores per employee.
- **`performance_history.csv`**: Employee training and performance scores.
- **`occupation_master.csv`**: O*NET role taxonomy metadata.
- **`essential_skills_processed.csv`**: Importance and Level scores for essential skills mapped to O*NET codes.
- **`software_skills_processed.csv`**: Software tools and software categories mapped to O*NET codes.
- **`role_skills.csv`**: Consolidated role-skills mapping joined from the O*NET taxonomy and skills tables.

---

## 2. Table-to-Table Relationships

### 2.1. `employees.csv` and `engagement_data.csv` / `performance_history.csv`
> [!IMPORTANT]
> **No Join Key (Independent Analytics Modules)**:
> As documented in the Project Key Decisions, these datasets originate from completely independent sources (public HR datasets) with different sizes (1,470 vs 2,843 rows) and have no shared identifiers.
> - `employees.csv` uses `EmployeeNumber` as its Primary Key.
> - `engagement_data.csv` and `performance_history.csv` use `Employee ID` as their Primary/Foreign Key.
> - **There is no relationship/join key between them**. They are kept as separate analytical tables and must NOT be merged.

### 2.2. `engagement_data.csv` and `performance_history.csv`
- **Relationship**: 1-to-1 matching on `Employee ID`.
- **Join Key**: `Employee ID`
- **Description**: Both datasets represent different attributes of the same cohort of 2,843 employees. They can be joined on `Employee ID` to analyze correlation between engagement metrics and training history/performance scores.

### 2.3. `occupation_master.csv` and `role_skills.csv`
- **Relationship**: 1-to-Many
- **Join Key**: `O*NET-SOC Code`
- **Description**: Each occupation in `occupation_master` has multiple corresponding skill mappings (both essential skills and software tools) in `role_skills`.

### 2.4. `essential_skills_processed.csv` / `software_skills_processed.csv` and `occupation_master.csv`
- **Relationship**: Many-to-1
- **Join Key**: `O*NET-SOC Code`
- **Description**: Raw O*NET skills tables map to job role titles and descriptions in `occupation_master` via `O*NET-SOC Code`.

---

## 3. Data Integration: `role_skills.csv` Build Plan

`role_skills.csv` was built in `notebooks/04_data_relationships.ipynb` using the following steps:
1. Filter `essential_skills_processed.csv` to keep only the `Importance` scale (exactly 10 essential skills per occupation).
2. Clean and format the columns of both essential skills and software skills to align their schemas:
   - For essential skills: map `Element Name` to `Skill Name` and keep `Data Value` as `Relevance Score`.
   - For software skills: map `Workplace Example` to `Skill Name`, and assign a `Relevance Score` on a 3.0 to 5.0 scale using: `3.0 + 1.0 (if Hot Technology == 'Y') + 1.0 (if In Demand == 'Y')`.
3. Concatenate the two skills collections.
4. Join the combined skills with `occupation_master.csv` on `O*NET-SOC Code` via an inner join.
